#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Mat srcImage, org, img, tmp;

vector<Point> point1, point2;   //分别存储两个端点
vector<double> k_value;         //指定直线的斜率

ofstream lane_data;

int imagenum = 0;

double calcudist(Point p, Point p1, Point p2) 
{ 
    //先算出三条边的长度a b c 
    double a,b,c; 
    double s;//面积 
    double hl;//周长的一半 
    double h;//距离 
    a=sqrt(abs(p.x- p1.x) * abs(p.x- p1.x) + abs(p.y- p1.y)*abs(p.y- p1.y)); 

    b=sqrt(abs(p.x- p2.x) * abs(p.x- p2.x) + abs(p.y- p2.y)*abs(p.y- p2.y)); 

    c=sqrt(abs( p1.x- p2.x)*abs(p1.x- p2.x) + abs( p1.y- p2.y)*abs(p1.y- p2.y)); 

    hl=(a+b+c)/2; 
    s=sqrt(hl*(hl-a)*(hl-b)*(hl-c)); 

    h=(2*s)/c; 
    return h; 
}

void on_mouse(int event,int x,int y,int flags,void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号  
{  
    static Point pre_pt(0, 0);//初始坐标  
    static Point cur_pt(0, 0);//实时坐标  
    if (event == EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆  
    {  
        pre_pt = Point(x,y);  
        circle(org,pre_pt, 2, Scalar(255,0,0,0), FILLED, LINE_AA,0);//划圆  
        imshow("img",org);  
    }  
    else if (event == EVENT_MOUSEMOVE && !(flags & EVENT_FLAG_LBUTTON))//左键没有按下的情况下鼠标移动的处理函数  
    {  
        cur_pt = Point(x,y);  
        imshow("img",org);  
    }  
    else if (event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON))//左键按下时，鼠标移动，则在图像上划矩形  
    {  
        org.copyTo(tmp);  
        cur_pt = Point(x,y);  
        line(tmp,pre_pt,cur_pt,Scalar(0,255,0),4,8,0);//在临时图像上实时显示鼠标拖动时形成的矩形  
        imshow("img",tmp);  
    }  
    else if (event == EVENT_LBUTTONUP)//左键松开，将在图像上划矩形  
    {  
        cur_pt = Point(x,y);  
        circle(org, pre_pt, 2, Scalar(255,0,0,0), FILLED, LINE_AA, 0);  
        line(org, pre_pt, cur_pt, Scalar(0,255,0), 4, 8, 0);//根据初始点和结束点，将矩形画到img上
        point1.push_back(pre_pt), point2.push_back(cur_pt);  
        imshow("img",org);    
        waitKey(0);  
    }  
}  
void setlanelines()     //标定车道线
{
    org = srcImage.clone();
    namedWindow("img", WINDOW_FULLSCREEN);//定义一个img窗口  
    setMouseCallback("img",on_mouse,0);//调用回调函数  
    imshow("img",srcImage); 
    waitKey(0);
    for(int i = 0 ; i < point1.size(); i++)
    {
        cout<<point1[i]<<" "<<point2[i]<<endl;
        double slope = (point1[i].x - point2[i].x)*1.0 / (point1[i].y - point2[i].y)*1.0; //斜率的计算公式为列序号的差值除以行序号的差值,以防止出现竖直线
        k_value.push_back(slope);
    }
}
void getlanelines()     //根据标定的车道线通过霍夫变换得到符合条件的线段
{
    org = srcImage.clone();
    Mat midImage = srcImage.clone();
    GaussianBlur(midImage , midImage , Size(5, 5), 0, 0);
    cvtColor(midImage, midImage, COLOR_BGR2GRAY);
    int blockSize = 5, constValue = 10;
	adaptiveThreshold(midImage, midImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, blockSize, constValue);
	Mat element = getStructuringElement(MORPH_RECT, Size(1, 1));
	erode(midImage, midImage, element);
    medianBlur(midImage,midImage,3);

    imshow("sobel", midImage);
    //waitKey(0);

    for(int i = 0; i<srcImage.cols; i++)    //对上半部分清零
    {
        for(int j=0; j<610;j++)
        {
            midImage.at<uchar>(j, i)=0;
        }
        for(int j=850; j<srcImage.rows;j++)//对车部分清零
        {
            midImage.at<uchar>(j, i)=0;
        }
    }

    vector<Vec4i> lines;
    //与HoughLines不同的是，HoughLinesP得到lines的是含有直线上点的坐标的，所以下面进行划线时就不再需要自己求出两个点来确定唯一的直线了
    HoughLinesP(midImage, lines, 1, CV_PI / 180, 20, 20, 5);//注意第五个参数，为阈值
    //lines = cv2.HoughLinesP(binary, 0.5, np.pi / 180, 20, None, 50, 150)
/*
    HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
    //dst:边缘检测的输出图像. 它应该是个灰度图,作为我们的输入图像
    //１：参数极径 r 以像素值为单位的分辨率. 我们使用 1 像素.
    //CV_PI/180:参数极角 \theta 以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
    //50:要”检测” 一条直线所需最少的的曲线交点
    //50:能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
    //10:线段上最近两点之间的阈值,也就是能被认为在一条直线上的亮点的最大距离.
*/

    for(int index = 0; index < k_value.size(); index++)             //求每条指定车道线的最符合线段
    {
        double min_slope_distance = 9999.0;
        double min_point_distance = 9999.0;
        int choosen_index = 0;
        double enlarge = 1000.0;

        Point central((point1[index].x + point2[index].x) / 2, (point1[index].y + point2[index].y) / 2);//标定的直线中心点
        for(size_t i = 0; i < lines.size(); i++)                    //寻找斜率最近的线段序号
        {
            Vec4i l = lines[i];
            //画出所有检测出的线段
            line(org, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 3, 8, 0); //Scalar函数用于调节线段颜色

            double slope = (l[0] - l[2])*1.0 / (l[1] - l[3])*1.0;   //斜率计算公式为列序号除以行序号差值
        
            double slope_distance = abs(k_value[index]*enlarge*1.0 - slope*enlarge*1.0);  //将斜率放大100倍,以减少误差

            double point_distance = calcudist(central, Point(l[0], l[1]), Point(l[2], l[3]));//模板直线中心点与检测直线的距离
            if(min_slope_distance*enlarge > slope_distance*enlarge && point_distance < 20)    //一层筛选,得到的线段不一定是同斜率下最近的线段
            {
                //cout<<"first"<<endl;
                min_slope_distance = slope_distance;
                min_point_distance = point_distance;
                choosen_index = i;
            }
        }
        /*for(size_t i = 0; i < lines.size(); i++)
        {
            Vec4i l = lines[i];
            
            double slope = (l[0] - l[2])*1.0 / (l[1] - l[3])*1.0;   //斜率计算公式为列序号除以行序号差值
        
            double slope_distance = abs(k_value[index]*enlarge*1.0 - slope*enlarge*1.0);  //将斜率放大100倍,以减少误差

            double point_distance = calcudist(central, Point(l[0], l[1]), Point(l[2], l[3]));//模板直线中心点与检测直线的距离
            if(slope_distance*enlarge < min_slope_distance*enlarge && min_point_distance > point_distance)    //二层筛选,得到的线段在最近斜率下的距离最近线段
            {
                cout<<"second"<<endl;
                min_point_distance = point_distance;
                choosen_index = i;
            }
        }*/
        cout<<"l: "<<index<<" slope_distance: "<<min_slope_distance<<" point_distance: "<<min_point_distance<<endl;
        double slope = (lines[choosen_index][0] - lines[choosen_index][2])*1.0 / (lines[choosen_index][1] - lines[choosen_index][3])*1.0;  //列序号差值除以行序号差值
        k_value[index] = slope;     //对斜率进行更新

        int up_x = srcImage.rows / 2 - lines[choosen_index][1];
        int up_y = cvRound(up_x*1.0 * slope) + lines[choosen_index][0];
        Point up(up_y, srcImage.rows / 2);

        int down_x = srcImage.rows-1 - lines[choosen_index][1];
        int down_y = cvRound(down_x*1.0 * slope) + lines[choosen_index][0];
        Point down(down_y, srcImage.rows-1);

        if(min_point_distance<5.0) //当线段偏移过大时采取之前帧检测到的线段, 根据先验经验,以10.0为阈值
        {
            point1[index] = up;         //对模板的端点进行更新
            point2[index] = down;
        }

        //line(srcImage, up, down, Scalar(186, 88, 255), 4, 8, 0); //Scalar函数用于调节线段颜色
        line(srcImage, point1[index], point2[index], Scalar(186, 88, 255), 4, 8, 0);
    }

    namedWindow("canny", WINDOW_NORMAL);
    namedWindow("result", WINDOW_FULLSCREEN);
    imshow("canny", org);
    imshow("result", srcImage);
}
void empty()        //对所有向量进行清空
{
    point1.resize(0);
    point2.resize(0);
    k_value.resize(0);
    cout<<point1.size()<<" "<<point2.size()<<" "<<k_value.size()<<endl;
}
void getparameter()      //得到直线一般式AX+BY+C=0
{
/*
    A = Y2 - Y1
    B = X1 - X2
    C = X2*Y1 - X1*Y2
*/
    lane_data.open("/home/judy/Documents/image_dateset/****/back/"+to_string(imagenum)+".txt", ios::trunc);
    lane_data << "AX+BY+C=0, X is the index of column, Y is the index of row"<<endl;
    for(int i = 0; i< point1.size(); i++)
    {
        int y1 = point1[i].y, y2 = point2[i].y, x1 = point1[i].x, x2 = point2[i].x;
        int A = y2 - y1;
        int B = x1 - x2;
        int C = x2*y1 - x1*y2;
        lane_data<<"lane"<<i<<": A: "<<A<<" B: "<<B<<" C: "<<C<<endl;
        //cout<<"imagenum: "<<imagenum<<endl;
        //cout<<"lane"<<i<<": A: "<<A<<" B: "<<B<<" C: "<<C<<endl;
    }
    lane_data.close();
}
int main()
{
	VideoCapture cap;
	cap.open("/home/judy/Documents/image_dateset/Jinan2Qingdao/IMG_0309.MOV");

    while(imagenum++ < 0)   //车道线检测起点
    cap.read(srcImage);

    cap.read(srcImage);     //读取第一帧以标注车道线

    setlanelines();         //手动标定车道线
    getparameter();
 
    /*******************以上步骤为从第一帧标注车道线***********************/

    while(cap.read(srcImage))
    {
        //if(imagenum==2853) break;
        imagenum++;
        cout<<imagenum<<endl;
        getlanelines();     //根据标定的车道线通过霍夫变换得到符合条件的线段

        char command = waitKey(0);
        if (command == 'r')
        {
            //cout<<"r: "<<imagenum<<endl;
            destroyAllWindows();
            empty();
            setlanelines();     //手动标定车道线
            getlanelines();     //根据标定的车道线通过霍夫变换得到符合条件的线段
        }
        getparameter();
        if (command == 'q')    
        {    
            break;    
        }    
    }
    return 0;
}
