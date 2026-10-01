import cv2

print("Starting camera test...")

for camera_index in [0, 1, 2]:
    print(f"\nTrying camera index: {camera_index}")

    camera = cv2.VideoCapture(camera_index, cv2.CAP_ANY)

    if camera.isOpened():
        print(f"SUCCESS! Camera opened at index {camera_index}")

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read frame.")
                break

            frame = cv2.flip(frame, 1)

            cv2.imshow("Camera Test", frame)

            key = cv2.waitKey(1)

            if key == 27:
                break

        camera.release()
        cv2.destroyAllWindows()

        print(f"\nWorking camera index: {camera_index}")
        break
    else:
        print(f"Camera index {camera_index} failed.")
        camera.release()
else:
    print("\nERROR: No camera could be opened.")