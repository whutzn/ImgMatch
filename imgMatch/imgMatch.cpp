// imgMatch.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void spatial_ostu(InputArray _src, int grid_x, int grid_y) {
	Mat src = _src.getMat();
	int width = src.cols / grid_x;
	int height = src.rows / grid_y;
	// iterate through grid
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
			cv::threshold(src_cell, src_cell, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
		}
	}
}

void imageSegment(){
	int value[7] = { 1, 20, 45, 64, 83, 101, 120 };
	vector<cv::Mat> resultVec;
	vector<cv::String> fn;
	glob("C:\\PlateOcr\\easyPR\\plate\\plate\\*.jpg",fn);
	for (size_t i = 0; i < fn.size(); i++)
	{
		Mat src = imread(fn[i]);
		Mat input_grey;
		cvtColor(src, input_grey, CV_BGR2GRAY);
		Mat img_threshold;
		img_threshold = input_grey.clone();
		spatial_ostu(img_threshold, 8, 2);
		for (size_t j = 0; j < 7; j++)
		{
			Mat tempsrc = img_threshold(Rect(value[j], 9, 15, 20));
			string imgname = "";
			imwrite(imgname, tempsrc); 
		}
	}
}

void test_knn(){
	Mat img = imread("digits.png");
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;   //原图为1000*2000
	int n = gray.cols / b;   //裁剪为5000个20*20的小图块
	Mat data, labels;   //特征矩阵
	for (int i = 0; i < n; i++)
	{
		int offsetCol = i*b; //列上的偏移量
		for (int j = 0; j < m; j++)
		{
			int offsetRow = j*b;  //行上的偏移量
			//截取20*20的小块
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			data.push_back(tmp.reshape(0, 1));  //序列化后放入特征矩阵
			labels.push_back((int)j / 5);  //对应的标注
		}

	}
	data.convertTo(data, CV_32F); //uchar型转换为cv_32f
	int samplesNum = data.rows;
	int trainNum = 3000;
	Mat trainData, trainLabels;
	trainData = data(Range(0, trainNum), Range::all());   //前3000个样本为训练数据
	trainLabels = labels(Range(0, trainNum), Range::all());

	//使用KNN算法
	int K = 4;
	Ptr<TrainData> tData = TrainData::create(trainData, ROW_SAMPLE, trainLabels);
	Ptr<KNearest> model = KNearest::create();
	model->setDefaultK(K);
	model->setIsClassifier(true);
	model->train(tData);

	//预测分类
	double train_hr = 0, test_hr = 0;
	Mat response;
	// compute prediction error on train and test data
	for (int i = 0; i < samplesNum; i++)
	{
		Mat sample = data.row(i);
		float r = model->predict(sample);   //对所有行进行预测
		//预测结果与原结果相比，相等为1，不等为0
		r = std::abs(r - labels.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

		if (i < trainNum)
			train_hr += r;  //累积正确数
		else
			test_hr += r;
	}

	test_hr /= samplesNum - trainNum;
	train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

	printf("accuracy: train = %.1f%%, test = %.1f%%\n",
		train_hr*100., test_hr*100.);
}

int _tmain(int argc, _TCHAR* argv[])
{
	imageSegment();
	system("pause");
	return 0;
}

