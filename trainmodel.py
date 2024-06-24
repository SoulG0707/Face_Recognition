import os
import face_recognition as fr
import pickle


def encode_faces(image_dir='./Images'):
    encoded_data = {}
    for dirpath, _, fnames in os.walk(image_dir):
        for fname in fnames:
            if fname.lower().endswith(('.jpg', '.png')):
                folder_name = os.path.basename(dirpath)
                image_path = os.path.join(dirpath, fname)
                face = fr.load_image_file(image_path)
                try:
                    encoding = fr.face_encodings(face)[0]
                    encoded_data.setdefault(folder_name, []).append(encoding)
                    print(f"Successfully encoded {fname} in {folder_name}")
                except IndexError:
                    print(f"Face not found in the image {fname}")
    return encoded_data


if __name__ == "__main__":
    model_file = 'face_recognition_model.dat'

    # Re-encode faces
    faces = encode_faces()

    # Save updated model
    with open(model_file, 'wb') as f:
        pickle.dump(faces, f)

    print("Training completed and data saved to face_recognition_model.dat")
