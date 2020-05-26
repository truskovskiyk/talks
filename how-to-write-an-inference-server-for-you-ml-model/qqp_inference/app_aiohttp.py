from aiohttp import web
from qqp_inference.model import PythonPredictor


async def handle_predict(request: web.Request) -> web.Response:
    predictor: PythonPredictor = request.app["predictor"]
    payload = await request.json()
    # This is cpu_bound operation, should be run with ProcessPoolExecutor
    result = predictor.predict(payload=payload)
    return web.json_response(result)


app = web.Application()
app["predictor"] = PythonPredictor.create_for_demo()
app.router.add_post("/predict", handle_predict)


if __name__ == "__main__":
    web.run_app(app)
