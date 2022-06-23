import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

class twoD_coordinates_GNB:
    def __init__(self):
        self.count_enter = 0
        self.which_label = 1

        self._background_figure()
        self.cid1 = self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid2 = self.figure.canvas.mpl_connect('key_press_event', self._on_press)

        self.labeled_coordinates = {}
        self.labeled_coordinates[-1] = []
        self.labeled_coordinates[1] = []
        plt.show()

    def give_labeled_coordinates(self):
        return self.labeled_coordinates

    def _on_click(self, event):
        if self.which_label == 1:
            self._draw_click(event)
            self.labeled_coordinates[1].append([event.xdata, event.ydata])
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {1}')
        else:
            self._draw_click(event)
            self.labeled_coordinates[-1].append(([event.xdata, event.ydata]))
            print(f'Coordinate: [{event.xdata}, {event.ydata}], Label: {-1}')

    def _on_press(self, event):
        if self._is_enter(event):
            if self.count_enter == 0:
                self.count_enter = 1
                self.which_label = 2

                self._background_figure()

            elif self.count_enter == 1:
                self.count_enter = 2
                self.figure.canvas.mpl_disconnect(self.cid1)

            else:
                self._linear_Gaussian_Naive_Bayes()
                self.figure.canvas.mpl_disconnect(self.cid2)

        else:
            print('You pressed a wrong button')

    def _linear_Gaussian_Naive_Bayes(self):
        mean_11 = np.mean(self.labeled_coordinates[1][0])
        mean_21 = np.mean(self.labeled_coordinates[1][1])
        mean_12 = np.mean(self.labeled_coordinates[-1][0])
        mean_22 = np.mean(self.labeled_coordinates[-1][1])

        # Under assumption that stds vary from feature to feature, but are the same regardless of y,
        # Gaussian Naive Bayes classifier has linear decision boundary.
        # So we calculate var of data points for each feature dimension, only given y=1
        var_1 = np.var(self.labeled_coordinates[1][0])
        var_2 = np.var(self.labeled_coordinates[1][1])

        y1_proba = len(self.labeled_coordinates[1]) / (len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))
        y2_proba = len(self.labeled_coordinates[-1]) / (len(self.labeled_coordinates[1]) + len(self.labeled_coordinates[-1]))

        x = np.array(range(-5, 6))
        y = [-var_2*(mean_22-mean_21) * ((mean_12-mean_21)/var_1*x_i + np.log((y1_proba/y2_proba)) +
                                        (mean_11**2 - mean_12**2)/(2*var_1) + (mean_21**2 - mean_22**2)/(2*var_2)) for x_i in x]

        plt.cla()
        self._background_figure()
        self._redraw_plots()
        plt.plot(x, y)
        plt.fill_between(x, -10, y, alpha=.25)
        plt.fill_between(x, y, 10, alpha=.25, color='r')

    def _background_figure(self):
        if self.count_enter == 0:
            self.figure, self.ax = plt.subplots()
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('click to plot from class 1. Please press enter when finished.')
        elif self.count_enter == 1:
            self.ax.set_title('click to plot from class 2. please double enter to start Naive Bayes.')
        else:
            self.ax.set_aspect(1)
            self.ax.set(xlim=[-5, 5], ylim=[-5, 5])
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('feature 1')
            self.ax.set_ylabel('feature 2')
            self.ax.set_title('Gaussian Naive Bayes')

    def _redraw_plots(self):
        for x, y in self.labeled_coordinates[1]:
            self.ax.scatter(x, y, marker='.', c='r')
        for x, y in self.labeled_coordinates[-1]:
            self.ax.scatter(x, y, marker='x', c='b')

    def _draw_click(self, event):
        if self.which_label == 1:
            self.ax.scatter(event.xdata, event.ydata, marker='.', c='r')
        else:
            self.ax.scatter(event.xdata, event.ydata, marker='x', c='b')

    def _is_enter(self, event):
        if event.key == 'enter':
            return True
        else:
            return False