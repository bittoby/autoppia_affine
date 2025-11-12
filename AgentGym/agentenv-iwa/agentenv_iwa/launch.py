import argparse
import uvicorn
from .server import app


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    launch()
