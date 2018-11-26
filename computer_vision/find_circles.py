import numpy as np
import cv2 as cv

class CVOperations(object):
    def __init__(self):
        pass

    def record_and_save(self, output_name='output'):
        cap = cv.VideoCapture(0)

        # Define the codec and create VideoWriter object.
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter('{}.avi'.format(output_name),
                fourcc,
                20.0,
                (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame - cv.flip(frame, 0)

                # Write the flipped frame.
                out.write(frame)

                cv.imshow('frame', frame)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished.
        cap.release()
        out.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    op = CVOperations()
    op.record_and_save()
