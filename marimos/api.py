import marimo
from fastapi import FastAPI

# Create a marimo asgi app
server = (
    marimo.create_asgi_app()
    .with_app(path="/", root="./marimos/network.py")
    .with_app(path="/network", root="./marimos/network.py")
    .with_app(path="/validation", root="./marimos/validation.py")
)

# Create a FastAPI app
app = FastAPI()

app.mount("/", server.build())

# Run the server using command-line:
# uvicorn marimos.api:app --host localhost --port 56000
# Run the file as script:
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
