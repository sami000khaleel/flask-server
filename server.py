from flask import Flask, request, jsonify
from similarities import go 
import os
from dotenv import load_dotenv
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity

load_dotenv()

# حفظ الملف المرفوع
def save_uploaded_file(file):
    os.makedirs('images', exist_ok=True)
    
    file_path = os.path.join('images', file.filename)
    file.save(file_path)
    return file_path

app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = os.getenv('SERVER_TOKEN_SECRET')
jwt = JWTManager(app)
@app.route('/', methods=['GET'])
def print_hello():
    return jsonify({'message':'helllo'}), 200

@app.route('/api/models/get-similarities', methods=['POST'])
@jwt_required()  # التأكد من وجود رمز مميز صالح
def get_similarities():
    try:
        print('has started psot')
        if 'file' not in request.files:
            return jsonify({'error': 'no file field was sent in the request'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'file was never specified'}), 400

        if file:
            file_path = save_uploaded_file(file)
            results = go(file_path)
            similarities = [[img[0], float(img[1])] for img in results]
            print(similarities)
            return jsonify({'similarities':similarities}), 200
        else:
            return jsonify({'error': 'an error occured'}), 500
    except Exception as e:
        print(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()