from services import app
from routers import chat, lessons, pdf, system, timer, voice

for route_module in (system, pdf, voice, chat, timer, lessons):
    app.include_router(route_module.router)
