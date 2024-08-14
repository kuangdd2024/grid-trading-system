from grid_trading.grid_handler import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host='0.0.0.0', port=8080)
