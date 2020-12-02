# tennis-analytica

This project will start by relying on Megan Fazio, Kyle Fisher, and Tori Fujinam's work at Stanford titled:
'Tennis Ball Tracking: 3D Trajectory Estimation using Smartphone Videos'

See here for their project report: https://web.stanford.edu/class/ee368/Project_Winter_1718/Reports/Fazio_Fisher_Fujinami.pdf

I first want to try and reconstruct the ball position and speed during a point. This requires tracking the ball, and knowing the court lines to estimate the ball position.


Todo list:

* Court line detection
* Camera calibration
* Ball detection
* Player position detection


Starting with the court lines. I use a filter for white on the image, and then use OpenCV's HoughLines to find the all the lines in the image.

![Alt text](saved_plots/best_selected.png?raw=true "Title")

![Alt text](saved_plots/rho_theta_plot.png?raw=true "Title")

I've picked the best lines (highlighted in green) by hand, but the goal is to automate this! That's the next step. Potentially using K-Means clustering and known horizontal/vertical angles of the court.


KMeans clustering doesn't do well at finding the exact lines I want - instead I may use it later for separating horizontal/vertical lines for intersections. For now I have used a specific thresholding for finding the lines (which may not be too bad if the angle of the camera stays constant - could just be taken as fixed viewing position). This has helped me to automatically select the lines of interest, and so it works on any frame like so:

![Alt text](saved_plots/best_selection-thresholding.png?raw=true "Title")

I can then solve the selected lines for the intersection points, given that $\rho$ and $\theta$ are known of each line.
The intersection works well:

![Alt text](saved_plots/intersections_found.png?raw=true "Title")

Next job is to try and calibrate the camera, or detect the ball.



