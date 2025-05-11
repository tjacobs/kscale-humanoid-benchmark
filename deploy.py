"""Module for deploying joystick-controlled policies on K-Bot."""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import colorlogging
import numpy as np
import pykos
import tensorflow as tf
from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from kscale.web.utils import get_robots_dir, should_refresh_file
from tabulate import tabulate
from xax.nn.geom import rotate_vector_by_quat

logger = logging.getLogger(__name__)

RunMode = Literal["real", "sim"]


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int  # nn_id is the index of the actuator in the neural network output
    kp: float
    kd: float
    max_torque: float
    joint_name: str


async def get_metadata(model_name_or_dir: str, cache: bool = True) -> list[Actuator]:
    # Assumes if the directory exists, it contains a metadata.json file
    if os.path.exists(Path(model_name_or_dir) / "metadata.json"):
        metadata_path = Path(model_name_or_dir) / "metadata.json"
    else:
        metadata_path = get_robots_dir() / model_name_or_dir / "metadata.json"

        if not cache or not (metadata_path.exists() and not should_refresh_file(metadata_path)):
            async with K() as api:
                robot_class = await api.get_robot_class(model_name_or_dir)
                if (metadata := robot_class.metadata) is None:
                    raise ValueError(f"No metadata found for {model_name_or_dir}")

            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)

    with open(metadata_path, "r") as f:
        metadata = RobotURDFMetadataOutput.model_validate_json(f.read())

    if metadata.joint_name_to_metadata is None:
        raise ValueError("Joint metadata is not available")

    joint_name_to_metadata = metadata.joint_name_to_metadata
    actuator_list = [
        Actuator(
            actuator_id=joint_metadata.id,
            nn_id=joint_metadata.nn_id,
            kp=float(joint_metadata.kp),
            kd=float(joint_metadata.kd),
            max_torque=float(joint_metadata.soft_torque_limit),
            joint_name=joint_name,
        )
        for joint_name, joint_metadata in joint_name_to_metadata.items()
        if joint_metadata.nn_id is not None
        and joint_metadata.id is not None
        and joint_metadata.kp is not None
        and joint_metadata.kd is not None
        and joint_metadata.soft_torque_limit is not None
    ]

    # Log the actuator configs
    table_data = [
        {
            "name": ac.joint_name,
            "id": ac.actuator_id,
            "nn_id": ac.nn_id,
            "kp": ac.kp,
            "kd": ac.kd,
            "max_torque": ac.max_torque,
        }
        for ac in sorted(actuator_list, key=lambda x: x.actuator_id)
    ]

    headers = {
        "name": "Joint Name",
        "id": "ID",
        "nn_id": "NN ID",
        "kp": "KP",
        "kd": "KD",
        "max_torque": "Max Torque",
    }

    table_string = tabulate(table_data, headers=headers, floatfmt=".2f")
    logger.info("Actuator configs:\n%s", table_string)

    return actuator_list


home_position = {
    21: 0.0,  # dof_right_shoulder_pitch_03
    22: -10.0,  # dof_right_shoulder_roll_03
    23: 0.0,  # dof_right_shoulder_yaw_02
    24: 90.0,  # dof_right_elbow_02
    25: 0.0,  # dof_right_wrist_00
    11: 0.0,  # dof_left_shoulder_pitch_03
    12: 10.0,  # dof_left_shoulder_roll_03
    13: 0.0,  # dof_left_shoulder_yaw_02
    14: -90.0,  # dof_left_elbow_02
    15: 0.0,  # dof_left_wrist_00
    41: -25.0,  # dof_right_hip_pitch_04
    42: 0.0,  # dof_right_hip_roll_03
    43: 0.0,  # dof_right_hip_yaw_03
    44: -50.0,  # dof_right_knee_04
    45: 25.0,  # dof_right_ankle_02
    31: 25.0,  # dof_left_hip_pitch_04
    32: 0.0,  # dof_left_hip_roll_03
    33: 0.0,  # dof_left_hip_yaw_03
    34: 50.0,  # dof_left_knee_04
    35: -25.0,  # dof_left_ankle_02
}


@dataclass
class DeployConfig:
    policy_path: str = field(default="", metadata={"help": "Path to the policy to deploy"})
    action_scale: float = field(default=0.1, metadata={"help": "Scale of the action outputs"})
    run_mode: RunMode = field(default="sim", metadata={"help": "Run mode"})
    joystick_enabled: bool = field(default=False, metadata={"help": "Whether to use joystick"})
    episode_length: int = field(default=10, metadata={"help": "Length of the episode to run in seconds"})
    ip: str = field(default="localhost", metadata={"help": "KOS server IP address"})
    port: int = field(default=50051, metadata={"help": "KOS server port"})
    metadata: str = field(default="kbot-v2", metadata={"help": "Metadata model / path to use for the policy"})
    cache: bool = field(default=True, metadata={"help": "Whether to use cached metadata"})
    # Logging
    debug: bool = field(default=False, metadata={"help": "Whether to run in debug mode"})
    log_dir: str = field(default="rollouts", metadata={"help": "Directory to save rollouts"})
    save_plots: bool = field(default=False, metadata={"help": "Whether to save plots"})
    # Policy parameters
    dt: float = field(default=0.02, metadata={"help": "Timestep of the policy"})
    rnn_carry_shape: tuple[int, int] = field(
        default=(5, 128), metadata={"help": "Shape of the RNN carry. (num_layers, hidden_size)"}
    )

    def __repr__(self) -> str:
        return "DeployConfig(\n" + "\n".join([f"  {k}={v}" for k, v in self.__dict__.items()]) + "\n)"

    def to_dict(self) -> dict:
        return self.__dict__


StepDataDictableKey: TypeAlias = Literal["obs", "cmd"]


class StepDataDict(TypedDict):
    action: list[float]
    obs: dict[str, list[float]]
    cmd: dict[str, list[float]]


class HeaderDict(TypedDict):
    units: dict[str, dict[str, str]]
    config: dict[str, dict]
    actuator_config: list[dict]
    home_position: dict[int, float]
    date: str


class RolloutDict(TypedDict):
    header: HeaderDict
    data: dict[str, StepDataDict]


async def run_policy(config: DeployConfig, actuator_list: list[Actuator]) -> None:
    async def get_obs(kos_client: pykos.KOS) -> dict:
        actuator_states, quaternion, imu_values = await asyncio.gather(
            kos_client.actuator.get_actuators_state([ac.actuator_id for ac in actuator_list]),
            kos_client.imu.get_quaternion(),
            kos_client.imu.get_imu_values(),
        )

        # Joint observations
        sorted_actuator_list = sorted(actuator_list, key=lambda x: x.nn_id)

        state_dict_pos = {state.actuator_id: state.position for state in actuator_states.states}
        pos_obs = [state_dict_pos[ac.actuator_id] for ac in sorted_actuator_list]
        pos_obs = np.deg2rad(np.array(pos_obs))  # PyKOS returns degrees, the model expects radians

        state_dict_vel = {state.actuator_id: state.velocity for state in actuator_states.states}
        vel_obs = np.deg2rad(np.array([state_dict_vel[ac.actuator_id] for ac in sorted_actuator_list]))

        # IMU observations
        projected_gravity = rotate_vector_by_quat(
            np.array([0, 0, -9.81]),  # type: ignore[arg-type]
            np.array([quaternion.w, quaternion.x, quaternion.y, quaternion.z]),  # type: ignore[arg-type]
            inverse=True,
        )

        imu_acc = np.array([imu_values.accel_x, imu_values.accel_y, imu_values.accel_z])
        imu_gyro = np.array([imu_values.gyro_x, imu_values.gyro_y, imu_values.gyro_z])

        return {
            "pos_obs": pos_obs,
            "vel_obs": vel_obs,
            "projected_gravity": projected_gravity,
            "imu_acc": imu_acc,
            "imu_gyro": imu_gyro,
        }

    def obs_to_vec(obs: dict, cmd: dict) -> np.ndarray:
        # Modify this as needed to match your model's input format
        return np.concatenate(
            [
                obs["pos_obs"],
                obs["vel_obs"],
                obs["projected_gravity"],
                obs["imu_acc"],
                obs["imu_gyro"],
            ]
        )[None, :]

    async def get_command(joystick_enabled: bool) -> dict:
        if joystick_enabled:
            raise NotImplementedError
        else:
            return {
                "linear_command_x": np.array([0.0]),
                "linear_command_y": np.array([0.0]),
                "angular_command": np.array([0.0]),
            }

    async def send_action(raw_action: np.ndarray, kos_client: pykos.KOS) -> None:
        raw_position = raw_action[: len(actuator_list)]

        position_target = np.rad2deg(raw_position) * config.action_scale

        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": ac.actuator_id,
                "position": position_target[ac.nn_id],
            }
            for ac in actuator_list
        ]

        await kos_client.actuator.command_actuators(actuator_commands)

    async def disable(kos_client: pykos.KOS) -> None:
        for ac in actuator_list:
            await kos_client.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=False,
                max_torque=ac.max_torque,
            )

    async def enable(kos_client: pykos.KOS) -> None:
        for ac in actuator_list:
            await kos_client.actuator.configure_actuator(
                actuator_id=ac.actuator_id,
                kp=ac.kp,
                kd=ac.kd,
                torque_enabled=True,
                max_torque=ac.max_torque,
            )

    async def go_home(kos_client: pykos.KOS) -> None:
        actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
            {
                "actuator_id": id,
                "position": position,
            }
            for id, position in home_position.items()
        ]

        await kos_client.actuator.command_actuators(actuator_commands)

    async def reset_sim(kos_client: pykos.KOS) -> None:
        logger.info("Resetting simulation...")
        await kos_client.sim.reset(
            pos={"x": 0.0, "y": 0.0, "z": 1.01},
            quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            joints=[{"name": ac.joint_name, "pos": home_position[ac.actuator_id]} for ac in actuator_list],
        )

    async def preflight(kos_client: pykos.KOS) -> None:
        os.makedirs(Path(config.log_dir) / config.run_mode, exist_ok=True)
        logger.info("Enabling motors...")
        await enable(kos_client)

        logger.info("Moving to home position...")
        await go_home(kos_client)

    async def postflight(timing_data: list[float]) -> None:
        datetime_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the directory for this specific rollout
        rollout_dir = Path(config.log_dir) / config.run_mode / datetime_name
        rollout_dir.mkdir(parents=True, exist_ok=True)

        rollout_file = rollout_dir / f"rollout_{datetime_name}.json"
        logger.info("Saving rollout data to %s", rollout_file)
        with open(rollout_file, "w") as f:
            json.dump(rollout_dict, f, indent=2)

        logger.info("Rollout data saved to %s", rollout_file)

        if timing_data:
            timings = np.array(timing_data)
            logger.info(
                "timing stats: min=%.6f, max=%.6f, median=%.6f seconds",
                np.min(timings),
                np.max(timings),
                np.median(timings),
            )
        else:
            logger.info("No timing data collected.")

        logger.info("Disabling motors...")
        await disable(kos_client)

        logger.info("Motors disabled")

        if config.save_plots:
            logger.info("Saving plots...")
            import matplotlib.pyplot as plt

            timestamps = [float(t) for t in rollout_dict["data"].keys()]
            data_values: list[StepDataDict] = list(rollout_dict["data"].values())

            plot_dir = rollout_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            def save_plot(
                filename_suffix: str, title: str, data_dict: StepDataDictableKey, labels: dict[str, str]
            ) -> None:
                plt.figure(figsize=(12, 6))
                for key, label in labels.items():
                    y_data = np.array([d[data_dict][key] for d in data_values])
                    if y_data.ndim == 1:
                        plt.plot(timestamps, y_data, label=label)
                    else:
                        for i in range(y_data.shape[1]):
                            plt.plot(timestamps, y_data[:, i], label=f"{label}_{i}")

                plt.title(title)
                plt.xlabel("Time (s)")
                plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                plt.grid(True)
                plt.tight_layout(rect=(0, 0, 0.9, 1))
                plot_path = plot_dir / f"{filename_suffix}_{datetime_name}.png"
                plt.savefig(plot_path)
                plt.close()
                logger.info("Plot saved to %s", plot_path)

            # Plot Observations
            save_plot("obs_pos", "Observed Joint Positions (rad)", "obs", {"pos_obs": "Position"})
            save_plot("obs_vel", "Observed Joint Velocities (rad/s)", "obs", {"vel_obs": "Velocity"})
            save_plot(
                "obs_imu",
                "Observed IMU Data",
                "obs",
                {"projected_gravity": "Gravity (m/s^2)", "imu_acc": "Acceleration (m/s^2)", "imu_gyro": "Gyro (rad/s)"},
            )

            # Plot Commands
            save_plot(
                "cmd",
                "Commands",
                "cmd",
                {
                    "linear_command_x": "Linear X (m/s)",
                    "linear_command_y": "Linear Y (m/s)",
                    "angular_command": "Angular (rad/s)",
                },
            )

            num_joints = len(actuator_list)

            action_data = np.array([d["action"] for d in data_values])
            pos_action = action_data[:, :num_joints]

            plt.figure(figsize=(12, 6))
            for i in range(pos_action.shape[1]):
                plt.plot(timestamps, pos_action[:, i], label=f"Pos Action Joint_{i}")
            plt.title("Action - Position Targets (rad)")
            plt.xlabel("Time (s)")
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout(rect=(0, 0, 0.9, 1))
            plot_path = plot_dir / f"action_pos_{datetime_name}.png"
            plt.savefig(plot_path)
            plt.close()
            logger.info("Plot saved to %s", plot_path)

    rollout_dict: RolloutDict = {
        "header": {
            "units": {
                "obs": {
                    "Projected gravity": "Units in m/s^2",
                    "Position": "Units in rad",
                    "Velocity": "Units in rad/s",
                },
                "cmd": {
                    "Linear command": "Units in m/s",
                    "Angular command": "Units in rad/s",
                },
                "action": {
                    "Position": "Units in rad",
                },
            },
            "config": config.to_dict(),
            "actuator_config": [ac.__dict__ for ac in actuator_list],
            "home_position": home_position,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": {},
    }

    timing_data: list[float] = []

    async with pykos.KOS(ip=config.ip, port=config.port) as kos_client:
        model = tf.saved_model.load(config.policy_path)

        # Warm up model
        logger.info("Warming up model...")
        obs = await get_obs(kos_client)
        cmd = await get_command(config.joystick_enabled)
        carry = np.zeros(config.rnn_carry_shape)[None, :]
        _ = model.infer(obs_to_vec(obs, cmd), carry)

        logger.info("Starting preflight...")
        await preflight(kos_client)

        logger.info("Press 'Enter' to start the rollout...")
        try:
            input()
        except Exception as e:
            logger.error("Error waiting for user input: %s", e)
            raise

        if config.run_mode == "real":
            for i in range(5, -1, -1):
                logger.info("Starting rollout in %d...", i)
                await asyncio.sleep(1)
        else:
            await reset_sim(kos_client)

        action_future: asyncio.Task | None = None

        start_time = time.monotonic()
        target_time = start_time + config.dt

        try:
            while time.monotonic() - start_time < config.episode_length:
                if action_future is not None and not action_future.done():
                    logger.info("Waiting for previous action to be transmitted...")
                    await action_future

                obs, cmd = await asyncio.gather(
                    get_obs(kos_client),
                    get_command(config.joystick_enabled),
                )

                action, carry = model.infer(obs_to_vec(obs, cmd), carry)

                action_array = np.array(action).reshape(-1)

                elapsed_time = time.monotonic() - start_time
                rollout_dict["data"][f"{elapsed_time:.4f}"] = StepDataDict(
                    obs={k: v.tolist() for k, v in obs.items()},
                    cmd={k: v.tolist() for k, v in cmd.items()},
                    action=action_array.tolist(),
                )

                # send_action_start = time.perf_counter()
                action_future = asyncio.create_task(send_action(action_array, kos_client))
                # await send_action(action_array, kos_client)
                # send_action_end = time.perf_counter()
                # timing_data.append(send_action_end - send_action_start)

                if time.monotonic() > target_time:
                    logger.warning("Loop overran by %f seconds", time.monotonic() - target_time)
                else:
                    logger.debug("Sleeping for %f seconds", target_time - time.monotonic())
                    await asyncio.sleep(max(0, target_time - time.monotonic()))

                target_time += config.dt

        except asyncio.CancelledError:
            logger.info("Episode cancelled")

        finally:
            await postflight(timing_data)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy_path", type=str)
    parser.add_argument("--action-scale", type=float, default=0.1)
    parser.add_argument("--run-mode", type=str, default="sim")
    parser.add_argument("--joystick-enabled", action="store_true")
    parser.add_argument("--episode-length", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--log-dir", type=str, default="rollouts")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--metadata", type=str, default="kbot")
    parser.add_argument("--cache", action="store_true")
    args = parser.parse_args()

    colorlogging.configure(level=logging.DEBUG if args.debug else logging.INFO)

    config = DeployConfig(**vars(args))

    logger.info("Args: %s", config)

    actuator_list = await get_metadata(model_name_or_dir=config.metadata, cache=config.cache)

    await run_policy(config, actuator_list)


if __name__ == "__main__":
    asyncio.run(main())
