import numpy as np
import matplotlib.pyplot as plt
import cv2

class CVOperations(object):
    def __init__(self, dp=1.2, min_dist=100, param_one=100, param_two=100, min_radius=0, max_radius=0, color_thresholds=np.array([100, 100, 100])):
        # See the documentation for OpenCV HoughCircles for an explanation of the parameters below.
        # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles#houghcircles
        self.hough_method = cv2.HOUGH_GRADIENT
        self.dp = dp
        self.min_dist = min_dist
        self.param_one = param_one
        self.param_two = param_two
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.color_thresholds = color_thresholds

    def set_dp(self, dp):
        self.dp = dp

    def draw_circle(self, circle, frame, color=(0, 255, 0)):
        x, y, r = circle
        cv2.circle(frame, (x, y), r, color, 4)
        cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


    def draw_circles_frame(self, circles, frame):
        if circles is not None:
            circles = np.round(circles[0, :]).astype('int')

            for circle in circles:
                self.draw_circle(circle, frame)

    def detect_circles_image(self, image_name='soccer.jpg'):
        image = cv2.imread(image_name)
        self.detect_circles_np_array(image)

    def most_likely_circle(self, circles, image):
        if circles is None:
            return None

        for circle in circles:
            circle = np.squeeze(circle)
            pixels = self.histogram_colors_in_circle(image, circle)
            if pixels:
                average_color = np.mean(np.array(pixels), axis=0)
                if (average_color > self.color_thresholds).all():
                #if average_color[-1] >= 100:
                    return circle.astype(np.int32)

        return None

    def histogram_colors_in_circle(self, image, circle):
        width, height, _ = image.shape
        x, y, r = circle
        pixels = []
        for i in range(0, width, 10):
            for j in range(0, height, 10):
                dx = i - x
                dy = j - y
                distance_squared = dx * dx + dy * dy
                if distance_squared <= r * r:
                    pixels.append(image[i][j])
        return pixels

    def detect_circles_np_array(self, image, output_name, wait=0):
        circle = None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist)
        self.draw_circles_frame(circles, image)
        if circles is not None:
            circles = np.squeeze(circles, axis=0)
            circle = self.most_likely_circle(circles, image)
            if circle is not None:
                self.draw_circle(circle, image, color=(0, 0, 255))
        cv2.imshow(output_name, image)
        cv2.waitKey(wait)

        return circle

    def detect_circles_video(self):
        """Detect circles in a video using Hough Circles.
        """
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist)
                self.draw_circles_frame(circles, frame)
                cv2.putText(frame,'OpenCV',(100,500), font, 4,(255,255,255),2,cv2.LINE_AA)
                if circles is not None:
                    circles = np.squeeze(circles, axis=0)
                    circle = self.most_likely_circle(circles, frame)
                    if circle is not None:
                        self.draw_circle(circle, frame, color=(255, 0, 0))
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
