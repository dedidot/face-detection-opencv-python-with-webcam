import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #hog

video = cv2.VideoCapture(0)
video.set(3,640)
video.set(4,480)

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    if name == '.DS_Store':
        continue
    
    #Load every file of faces of known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        if filename == '.DS_Store':
            continue
        
        image = face_recognition.load_image_file(f'{KNOWN_FACES_DIR}/{name}/{filename}')

        # Get 128-dimention face encoding
        encoding = face_recognition.face_encodings(image)[0]

        # Append encoding and image name
        known_faces.append(encoding)
        known_names.append(name)

print("Processing unknown faces....")

while True:
    ret, image = video.read()
    # Grab face location
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    print(f' found {len(encodings)} face(s) ')

    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        # Check unknown faces with known faces
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f' - {match} from {results} ')

            # Each location contains position in order: top, right, bottom, left
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            # Get color by name using our fancy function
            color = [0, 255, 0]#name_to_color(match)
            
            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)

            # Now we need smaller, filled grame bellow for a name
            # This time we use bottom in both corners - to start from bottom and moce 50 pixels down
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            # Paint frame
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            # Write a name
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
