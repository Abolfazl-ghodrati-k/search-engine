from flask import Flask
from controllers.product_controller import product_blueprint

app = Flask(__name__)
app.register_blueprint(product_blueprint)

if __name__ == "__main__":
    app.run(debug=True)
