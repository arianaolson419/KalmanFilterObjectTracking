import numpy as np
import matplotlib.pyplot as plt
import cv2

class CVOperations(object):
    def __init__(self, dp=1.2, min_dist=100, param_one=100, param_two=100, min_radius=0, max_radius=0):
        self.hough_method = cv2.HOUGH_GRADIENT
        self.dp = dp
        self.min_dist = min_dist
        self.param_one = param_one
        self.param_two = param_two
        self.min_radius = min_radius
        self.max_radius = max_radius

    def update_parameters(self, dp, min_dist, param_one, param_two, min_radius, max_radius):
        self.dp = dp
        self.min_dist = min_dist
        self.param_one = param_one
        self.param_two = param_two
        self.min_radius = min_radius
        self.max_radius = max_radius
        
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

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist)
        ouptut = self.draw_circles_image(circles, output)
        cv2.imshow('output', np.hstack([image, output]))
        cv2.waitKey(0)

    def detect_circles_image(self, image_name='soccer.jpg'):
        image = cv2.imread(image_name)
        self.detect_circles_np_array(image)

    def most_likely_circle(self, circles):
        pass

    def histogram_colors_in_circle(self, image, circle):
        width, height, _ = image.shape
        x, y, r = circle
        pixels = []
        for i in range(width):
            for j in range(height):
                dx = i - x
                dy = j - y
                distance_squared = dx * dx + dy * dy
                if distance_squared <= r * r:
                    pixels.append(image[i][j])
        print(np.mean(np.array(pixels), axis=0))
#        blue = [pixel[0] for pixel in pixels]
#        green = [pixel[1] for pixel in pixels]
#        red = [pixel[2] for pixel in pixels]
        
        return pixels

    def detect_circles_video(self):
        """Detect circles in a video using Hough Circles.
        """
        cap = cv2.VideoCapture(0)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.dp, self.min_dist)
                self.draw_circles_frame(circles, frame)
                if circles is not None:
                    self.histogram_colors_in_circle(frame, circles[0][0])
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
