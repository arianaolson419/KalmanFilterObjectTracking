import numpy as np
import cv2

class CVOperations(object):
    def __init__(self):
        pass

    # TODO: delete this function and put image drawing back into detect_circles_image.
    def draw_circles_image(self, circles, output):
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')

            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        return output

    def draw_circles_frame(self, circles, frame):
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')

            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
    def detect_circles_np_array(self, image):
        output = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
        ouptut = self.draw_circles_image(circles, output)
        cv2.imshow('output', np.hstack([image, output]))
        cv2.waitKey(0)

    def detect_circles_image(self, image_name='soccer.jpg'):
        image = cv2.imread(image_name)
        self.detect_circles_np_array(image)

    def detect_circles_video(self):
        """Detect circles in a video using Hough Circles.
        """
        cap = cv2.VideoCapture(0)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
                self.draw_circles_frame(circles, frame)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished.
        cap.release()
        cv2.destroyAllWindows()


    def record_and_save(self, output_name='output'):
        """ Records a video using the webcam and saves it as a .avi file.
        
        Parameters
        ----------
        output_name: the name of the saved video. Do not include '.avi' in the
            name. Defaults to 'output'.
        """
        cap = cv2.VideoCapture(0)

        # Define the codec and create VideoWriter object.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('{}.avi'.format(output_name),
                fourcc,
                20.0,
                (640,480))

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame - cv2.flip(frame, 0)

                # Write the flipped frame.
                out.write(frame)

                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished.
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    op = CVOperations()
    op.detect_circles_video()
