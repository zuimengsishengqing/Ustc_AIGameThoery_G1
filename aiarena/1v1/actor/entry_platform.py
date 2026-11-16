# 平台维度 actor 入口：与 entry.py 并列，仅在 config 处切换到 config_platform，
# 并将 get_model_class 替换为平台模型。复用 entry 的 run/_run 全部逻辑。
import os

from absl import app as absl_app
from absl import flags

import entry
import model_platform

from hok.hok1v1.env1v1 import interface_default_config

# 关键：entry._run 内部通过模块全局名 get_model_class 调用，这里替换为平台模型
entry.get_model_class = model_platform.get_model_class


def main(_):
    FLAGS = flags.FLAGS
    mem_pool_addr_list = FLAGS.mem_pool_addr.strip().split(";")
    monitor_ip = mem_pool_addr_list[0].split(":")[0]
    monitor_server_addr = f"{monitor_ip}:8086"
    from common.config_platform import Config

    entry.run(
        FLAGS.actor_id,
        FLAGS.config_path,
        FLAGS.model_pool_addr,
        FLAGS.single_test,
        FLAGS.port_begin,
        FLAGS.gc_server_addr,
        FLAGS.gamecore_req_timeout,
        FLAGS.max_frame_num,
        FLAGS.runtime_id_prefix,
        FLAGS.aiserver_ip,
        mem_pool_addr_list,
        FLAGS.max_episode,
        monitor_server_addr,
        Config,
        debug_log=FLAGS.debug_log,
    )


if __name__ == "__main__":
    flags.DEFINE_integer("actor_id", 0, "actor id")
    flags.DEFINE_string("mem_pool_addr", "localhost:35200", "address of memory pool")
    flags.DEFINE_string("model_pool_addr", "localhost:10016", "address of model pool")

    flags.DEFINE_boolean("single_test", 0, "test_mode")
    flags.DEFINE_string(
        "config_path",
        os.getenv("INTERFACE_CONFIG_PATH", interface_default_config),
        "config file for interface",
    )
    flags.DEFINE_integer(
        "gamecore_req_timeout",
        30000,
        "millisecond timeout for gamecore to wait reply from server",
    )

    flags.DEFINE_string(
        "gc_server_addr",
        os.getenv("GAMECORE_SERVER_ADDR", "127.0.0.1:23432"),
        "address of gamecore server",
    )
    flags.DEFINE_string(
        "aiserver_ip", os.getenv("AI_SERVER_ADDR", "127.0.0.1"), "the actor ip"
    )

    flags.DEFINE_integer("max_episode", -1, "max number for run episode")
    flags.DEFINE_string("monitor_server_addr", "127.0.0.1:8086", "monitor server addr")

    flags.DEFINE_string(
        "runtime_id_prefix",
        os.getenv("ACTOR_RUNTIME_ID_PREFIX", "actor-1v1"),
        "must not contain '_'",
    )
    flags.DEFINE_integer("port_begin", int(os.getenv("ACTOR_PORT_BEGIN", "35300")), "")
    flags.DEFINE_integer("max_frame_num", int(os.getenv("MAX_FRAME_NUM", "20000")), "")
    flags.DEFINE_boolean("debug_log", False, "use debug log level")
    absl_app.run(main)