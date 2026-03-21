#!/usr/bin/env python3
"""
打印 PM_v2（serial_pm_v2.xml）MuJoCo body id ↔ 名称，与 policy_switch.csv 中
body_pos_w_{id}_* / body_quat_w_{id}_* 的编号一致。

依赖: pip install mujoco

说明: 主模型里若含无效 keyframe，可先仅编译机器人子文件（与本脚本一致）；
      仿真中 body 顺序以当前加载的 mjModel 为准，一般与此表一致。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
_DEFAULT_XML = _REPO / "src/simulation/mujoco/assets/resource/robot/pm_v2/xml/serial_pm_v2.xml"


def main() -> int:
    try:
        import mujoco as mj
    except ImportError:
        print("请先安装: pip install mujoco", file=sys.stderr)
        return 1

    xml_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _DEFAULT_XML
    if not xml_path.is_file():
        print(f"找不到 XML: {xml_path}", file=sys.stderr)
        return 1

    text = xml_path.read_text(encoding="utf-8")
    text = re.sub(r"<keyframe>[\s\S]*?</keyframe>", "", text)
    # 相对 include（serial_links.xml 等）相对本文件目录解析
    import os

    old = os.getcwd()
    try:
        os.chdir(xml_path.parent)
        mj_model = mj.MjModel.from_xml_string(text)
    finally:
        os.chdir(old)

    n = mj_model.nbody
    print(f"# file: {xml_path}")
    print(f"# nbody = {n}")
    print("id\tname")
    for bi in range(n):
        name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, bi)
        print(f"{bi}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
