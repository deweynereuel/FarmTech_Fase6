"""Lista índices de câmeras disponíveis para localizar a webcam virtual Iriun."""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Procura câmeras acessíveis pelo OpenCV.")
    parser.add_argument("--max-index", type=int, default=5,
                        help="Último índice a testar (padrão: 5).")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf", "any"), default="auto",
                        help="Backend de captura do OpenCV no Windows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "OpenCV não está instalado. Execute: pip install -r ir_alem_webcam/requirements.txt"
        ) from error

    backend_map = {
        "auto": cv2.CAP_DSHOW if sys.platform.startswith("win") else None,
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": None,
    }
    backend = backend_map[args.backend]
    found = []
    for index in range(args.max_index + 1):
        camera = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if camera.isOpened():
            ok, frame = camera.read()
            if ok:
                height, width = frame.shape[:2]
                found.append(index)
                print(f"Câmera disponível: índice {index} ({width}x{height})")
            camera.release()

    if not found:
        print("Nenhuma câmera foi aberta. Confirme que o Iriun está conectado e teste --backend msmf.")
    else:
        print(f"Use o índice desejado no detector: --camera {found[0]}")


if __name__ == "__main__":
    main()
