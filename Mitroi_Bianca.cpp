#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <fstream>
// biblioteci lab5
#include <queue>
#include <stack>
#include <random>

#include <opencv2\opencv.hpp>

using namespace cv;

using namespace std;

std::vector<int> lab3_calchist(Mat_ <uchar> img) {
	std::vector<int> hist(256);
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			hist[img(i, j)]++;
	return hist;
}

Mat_ <uchar> lab2_binarization(Mat_ <uchar> img, int prag) {

	Mat_ <uchar> newimg(img.rows, img.cols);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) > prag)
				newimg(i, j) = 255;
			else
				newimg(i, j) = 0;
		}
	}
	return newimg;
}

bool lab2_isInside(Mat image, int i, int j) {
	if (i >= 0 && i < image.rows) {
		if (j >= 0 && j < image.cols)
			return true;
	}
	return false;
}

Mat_ <uchar> lab2_toGray(String filename) {
	Mat_ <Vec3b> img = imread(filename, IMREAD_COLOR);
	Mat_ <uchar> Gray(img.rows, img.cols);
	Gray.setTo(0);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Gray(i, j) = (img(i, j)[2] + img(i, j)[1] + img(i, j)[0]) / 3;
		}
	}
	return Gray;
}

float lab8_pragAutomat(Mat_<uchar> img, vector<int> hist) {
	float Imin = 0, Imax = 0;

	for (int i = 0; i < hist.size(); i++) {
		if (hist[i] != 0) {
			Imin = i;
			break;
		}
	}
	for (int i = 255; i >= 0; i--) {
		if (hist[i] != 0) {
			Imax = i;
			break;
		}
	}
	// cout << Imin << ' ' << Imax << endl;

	// loop
	float error = 0.1f;
	float currentPrag = (Imin + Imax) / 2, prevPrag = 0;
	do {
		prevPrag = currentPrag;
		float medieMin = 0, medieMax = 0, ariaMin = 0, ariaMax = 0;
		for (int i = 0; i < hist.size(); i++)
			if (i < prevPrag) {
				medieMin += i * hist[i];
				ariaMin += hist[i];
			}
			else {
				medieMax += i * hist[i];
				ariaMax += hist[i];
			}
		medieMin /= ariaMin;
		medieMax /= ariaMax;
		currentPrag = (medieMin + medieMax) / 2;
	} while (abs(currentPrag - prevPrag) > error);

	return currentPrag;
}

Mat_ <uchar> createTemplate(int x, int u) {
	Mat_ <uchar> templ(4 * u + 3 * x, 4 * u + 3 * x);
	templ.setTo(0);
	int uCoefi = 1, xCoefi = 0;
	bool incrementIx = true, incrementJx = true, row = false, col = false;
	for (int i = 0; i < templ.rows; i++) {
		int uCoefj = 1, xCoefj = 0;
		for (int j = 0; j < templ.cols; j++) {
			if (i == uCoefi * u + xCoefi * x) {
				incrementIx ? xCoefi++ : uCoefi++;
				incrementIx = !incrementIx;
				uCoefj = 1; xCoefj = 0;
				row = !row;
			}
			if (j == uCoefj * u + xCoefj * x) {
				incrementJx ? xCoefj++ : uCoefj++;
				incrementJx = !incrementJx;
				col = !col;
			}
			(row && col) ? templ(i, j) = 0 : templ(i, j) = 255;
		}
	}	
	return templ;
}

typedef struct myStr {
	Mat_ <uchar> img;
	int whitePoints;
}myStr;

myStr countWhitePixels(Mat_ <uchar> binarized, Mat_ <uchar> templ, int minX, int searchi, int searchj) {
	myStr str;
	Mat_ <uchar> newimg = binarized.clone();
	int whitePixels = 0;
	for (int i = 0; i < minX; i++)
		for (int j = 0; j < minX; j++) {
			if (binarized(i + searchi, j + searchj) == 255 && binarized(i + searchi, j + searchj) == templ(i, j))
				whitePixels++;
			if (templ(i, j) == 255)
				newimg(i + searchi, j + searchj) = 255;
		}
	str.img = newimg;
	str.whitePoints = whitePixels;
	return str;
}

String color_detect(Vec3b pixel) {

	int hue = pixel[0];
	int saturation = pixel[1];
	int value = pixel[2];

	// Define color ranges
	int red_low1 = 0, red_high1 = 10;
	int red_low2 = 170, red_high2 = 180;
	int orange_low = 11, orange_high = 25;
	int yellow_low = 26, yellow_high = 35;
	int green_low = 36, green_high = 70;
	int blue_low = 100, blue_high = 130;
	int white_low = 0, white_high = 180;

	// Check for white color first
	if (saturation < 30 && value > 190)
        return "White";
    // Proceed with other color detections
    else if ((hue >= red_low1 && hue <= red_high1) || (hue >= red_low2 && hue <= red_high2))
        return "Red";
    else if (hue >= orange_low && hue <= orange_high)
        return "Orange";
    else if (hue >= yellow_low && hue <= yellow_high)
        return "Yellow";
    else if (hue >= green_low && hue <= green_high)
        return "Green";
    else if (hue >= blue_low && hue <= blue_high)
        return "Blue";
    else
        return "Unknown";
}

void mask(int msi, int msj, int mgrid, int x, int u) {
	Mat_ <Vec3b> templ = imread("projectPhotos/template.png", IMREAD_COLOR);
	Mat_ <Vec3b> img = imread("projectPhotos/frame_0.png", IMREAD_COLOR);
	Mat_ <Vec3b> hsvimg;
	cvtColor(img, hsvimg, COLOR_BGR2HSV);
	Mat_ <Vec3b> masked(mgrid, mgrid);
	Vec3b square(0, 0, 0);
	int count = 1;
	for (int i = 0; i < mgrid; i++) {
		for (int j = 0; j < mgrid; j++) {
			if (templ(i + msi, j + msj) == Vec3b(0, 0, 0)) {
				masked(i, j) = hsvimg(i + msi, j + msj);
			}
			else {
				masked(i, j) = templ(i + msi, j + msj);
			}
		}
	}
	cout << color_detect(masked(u + x / 2, u + x / 2)) << ' ' << color_detect(masked(u + x / 2, 2 * u + x + x / 2)) << ' ' << color_detect(masked(u + x / 2, 3 * u + 2 * x + x / 2)) << endl;
	cout << color_detect(masked(2 * u + x + x / 2, u + x / 2)) << ' ' << color_detect(masked(2 * u + x + x / 2, 2 * u + x + x / 2)) << ' ' << color_detect(masked(2 * u + x + x / 2, 3 * u + 2 * x + x / 2)) << endl;
	cout << color_detect(masked(3 * u + 2 * x + x / 2, u + x / 2)) << ' ' << color_detect(masked(3 * u + 2 * x + x / 2, 2 * u + x + x / 2)) << ' ' << color_detect(masked(3 * u + 2 * x + x / 2, 3 * u + 2 * x + x / 2)) << endl;
	cout << masked(u + x / 2, u + x / 2) << ' ' << masked(u + x / 2, 2 * u + x + x / 2) << ' ' << masked(u + x / 2, 3 * u + 2 * x + x / 2) << endl;
	cout << masked(2 * u + x + x / 2, u + x / 2) << ' ' << masked(2 * u + x + x / 2, 2 * u + x + x / 2) << ' ' << masked(2 * u + x + x / 2, 3 * u + 2 * x + x / 2) << endl;
	cout << masked(3 * u + 2 * x + x / 2, u + x / 2) << ' ' << masked(3 * u + 2 * x + x / 2, 2 * u + x + x / 2) << ' ' << masked(3 * u + 2 * x + x / 2, 3 * u + 2 * x + x / 2) << endl;

	imwrite("projectPhotos/masked.png", masked);
	imshow("masked", masked);
	waitKey();
}

void continueing() {
	Mat_ <uchar> binarized = lab2_toGray("projectPhotos/binarized.png");

	int u = 20;
	int minX = binarized.rows - u;
	int x = (minX - 4 * u) / 3 + 1;

	Mat_ <uchar> templ = createTemplate(x, u);
	myStr str = countWhitePixels(binarized, templ, minX, 0, 0);
	int maxWhitePixels = str.whitePoints;
	int maxSearchi = 0, maxSearchj = 0, maxX = x, maxU = u, maxminX = 0;
	Mat_ <uchar> newimg = str.img;
	
	do {

		Mat_ <uchar> templ = createTemplate(x, u);

		for (int searchi = 0; searchi + minX < binarized.rows; searchi += u - 10)
			for (int searchj = 0; searchj + minX < binarized.cols; searchj += u - 10) {
				myStr result = countWhitePixels(binarized, templ, minX, searchi, searchj);
				if (result.whitePoints > maxWhitePixels) {
					maxWhitePixels = result.whitePoints;
					maxSearchi = searchi;
					maxSearchj = searchj;
					maxminX = minX;
					/*cout << maxWhitePixels << endl;*/
					//imshow("new", result.img);
					//waitKey();
				}
			}
		x -= 10;
		minX = 3 * x + 4 * u;
	} while (x > u);
	x = (maxminX - 4 * (u - 5)) / 3 + 1;
	templ = createTemplate(x, u - 5);
	myStr result = countWhitePixels(binarized, templ, maxminX, maxSearchi, maxSearchj);
	// imshow("new", result.img);
	// waitKey();
	// cout << maxSearchi << ' ' << maxSearchj << ' ' << x;
	imwrite("projectPhotos/template.png", result.img);
	mask(maxSearchi, maxSearchj, maxminX, x, u - 5);
}

void developing() {
	// pass to grayscale
	Mat_ <uchar> gray = lab2_toGray("projectPhotos/frame_0.png");
	// imshow("gray", gray);

	// denoising
	Mat_ <uchar> denoised;
	fastNlMeansDenoising(gray, denoised, 3.0f, 7, 21);
	// imshow("denoised", denoised);

	// edges extraction
	Mat_ <uchar> edges;
	Laplacian(denoised, edges, CV_8U, 3);
	// imshow("edges", edges);

	vector<int> hist = lab3_calchist(edges);

	// binarization
	Mat_ <uchar> binarized;
	int prag = (int)lab8_pragAutomat(edges, hist);
	binarized = lab2_binarization(edges, prag);
	// imshow("binarized", binarized);
	imwrite("projectPhotos/binarized.png", binarized);
	continueing();
}

void proiect() {

	Mat_ <Vec3b> image;
	namedWindow("Display window");
	VideoCapture cap(0);

	if (!cap.isOpened())
		cout << "cannot open camera";
	int key = 0;
	while (true) {
		cap >> image;
		imshow("Display window", image);
		key = waitKey(1);
		if (key == ' ') {
			String filename = "projectPhotos/frame_0.png";
			imwrite(filename, image);
			developing();
		}
		if (key == 27)
			break;
	}
}

int main() {

	proiect();

	return 0;
}