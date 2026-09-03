"""命令行入口：单张图片或目录批量预测。

用法示例:
    python -m scripts.predict_cli sample_data/test.jpg     # 单图，JSON 打印到终端
    python -m scripts.predict_cli sample_data              # 目录批量，结果保存到 outputs/
    python -m scripts.predict_cli sample_data --conf 0.5 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from app.core.config import Settings
from app.vision.model_loader import ModelNotFoundError
from app.vision.predictor import InvalidImageError, predict_directory, predict_image


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 视觉检测")
    parser.add_argument("path", help="图片文件路径或图片目录")
    parser.add_argument("--model", help="模型权重路径（默认取配置）")
    parser.add_argument("--conf", type=float, help="置信度阈值（默认取配置）")
    parser.add_argument("--device", help="推理设备，如 cpu / 0（默认取配置）")
    parser.add_argument("--imgsz", type=int, help="推理输入尺寸（默认取配置）")
    parser.add_argument("--output", help="JSON 结果保存路径（默认自动生成）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = Settings.from_env()
    if args.model:
        settings = replace(settings, model_path=Path(args.model))
    if args.conf is not None:
        settings = replace(settings, confidence_threshold=args.conf)
    if args.device:
        settings = replace(settings, device=args.device)
    if args.imgsz:
        settings = replace(settings, image_size=args.imgsz)

    source = Path(args.path)
    try:
        if source.is_dir():
            results = predict_directory(source, settings=settings)
            payload = {name: result.to_dict() for name, result in results.items()}
            out_path = Path(args.output) if args.output else settings.output_dir / (
                f"predictions_{datetime.now():%Y%m%d_%H%M%S}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            total = sum(len(result.detections) for result in results.values())
            print(f"批量预测完成: {len(results)} 张图片, 共 {total} 个目标 -> {out_path}")
            for name, result in results.items():
                print(f"  {name}: {len(result.detections)} 个目标")
        else:
            result = predict_image(source, settings=settings)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"结果已保存: {out_path}", file=sys.stderr)
    except (InvalidImageError, ModelNotFoundError, OSError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
