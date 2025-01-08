#ifndef IMAGEPREPROCESSING_HPP
#define IMAGEPREPROCESSING_HPP

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <cassert>

namespace ImgProc {
    // 灰度图像
    using Img_Gray = std::vector<std::vector<unsigned char>>;
    
    // RGB 图像
    using Img_RGB = std::vector<std::vector<std::vector<unsigned char>>>;
    
    // RGBA 图像
    using Img_RGBA = std::vector<std::vector<std::vector<unsigned char>>>;
    
    // 多通道图像（动态通道数）
    using Img_MultiChannel = std::vector<std::vector<std::vector<unsigned char>>>;
    
    // 二值图像（使用 unsigned char 表示 0 或 255）
    using Img_Binary = std::vector<std::vector<unsigned char>>;
    
    // 浮点灰度图像
    using Img_Float = std::vector<std::vector<float>>;
    
    // 灰度浮点图像（仅浮点值）
    using Img_GrayFloat = std::vector<std::vector<float>>;
    
    // RGB 浮点图像（每个通道为浮点值）
    using Img_RGBFloat = std::vector<std::vector<std::vector<float>>>;

    // 单通道图像（仅一个通道）
    using SingleChannelImage = std::vector<std::vector<float>>;

    // 多通道图像（多个通道）
    using MultiChannelImage = std::vector<std::vector<std::vector<float>>>;

    namespace Sobel {
        std::vector<std::vector<int>> Gx{
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        std::vector<std::vector<int>> Gy{
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        };
    }

    namespace Laplacian {
        std::vector<std::vector<int>> LaplacianEdgeDetection{
            {0,-1,0},
            {-1,4,-1},
            {0,-1,0}
        };
    }

    const std::vector<std::vector<float>> test_3x3{
        {1,1,1},
        {1,1,1},
        {1,1,1}
    };
    const std::vector<std::vector<float>> test_5x5{
        {1,1,1,1,1},
        {1,1,1,1,1},
        {1,1,1,1,1},
        {1,1,1,1,1},
        {1,1,1,1,1}
    };

    // 均值模糊核
    const std::vector<std::vector<float>> AverageBlurKernel_3x3{
        {0.111f, 0.111f, 0.111f},
        {0.111f, 0.111f, 0.111f},
        {0.111f, 0.111f, 0.111f}
    };
    const std::vector<std::vector<float>> AverageBlurKernel_5x5{
        {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
        {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
        {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
        {0.04f, 0.04f, 0.04f, 0.04f, 0.04f},
        {0.04f, 0.04f, 0.04f, 0.04f, 0.04f}
    };

    // 高斯模糊核
    const std::vector<std::vector<float>> GaussianBlurKernel_3x3{
        {0.0625f, 0.125f, 0.0625f},
        {0.125f, 0.25f, 0.125f},
        {0.0625f, 0.125f, 0.0625f}
    };
    const std::vector<std::vector<float>> GaussianBlurKernel_5x5{
        {0.003f, 0.013f, 0.022f, 0.013f, 0.003f},
        {0.013f, 0.059f, 0.097f, 0.059f, 0.013f},
        {0.022f, 0.097f, 0.162f, 0.097f, 0.022f},
        {0.013f, 0.059f, 0.097f, 0.059f, 0.013f},
        {0.003f, 0.013f, 0.022f, 0.013f, 0.003f}
    };

    // 运动模糊核
    // 水平方向
    const std::vector<std::vector<float>> MotionBlurKernel_Horizontal_3x3{
        {0.0f, 0.0f, 0.0f},
        {0.333f, 0.333f, 0.333f},
        {0.0f, 0.0f, 0.0f}
    };
    const std::vector<std::vector<float>> MotionBlurKernel_Horizontal_5x5{
        {0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.2f,0.2f,0.2f,0.2f,0.2f},
        {0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,0.0f},
        {0.0f,0.0f,0.0f,0.0f,0.0f},
    };
    // 垂直方向
    const std::vector<std::vector<float>> MotionBlurKernel_Vertical_3x3{
        {0.0f, 0.333f, 0.0f},
        {0.0f, 0.333f, 0.0f},
        {0.0f, 0.333f, 0.0f}
    };
    const std::vector<std::vector<float>> MotionBlurKernel_Vertical_5x5{
        {0.0f,0.2f,0.0f,0.0f,0.0f},
        {0.0f,0.2f,0.0f,0.0f,0.0f},
        {0.0f,0.2f,0.0f,0.0f,0.0f},
        {0.0f,0.2f,0.0f,0.0f,0.0f},
        {0.0f,0.2f,0.0f,0.0f,0.0f},
    };
}

namespace ImgPreprocessing {

#define M_PI 3.14159265358979323846

    class ImagePreprocess {
        /*
        * 图像预处理类
        * 功能：
        * 1. 图像调整大小,按照指定宽高进行调整  按比例缩放
        * 2. 图像灰度化
        * 3. 图像二值化
        * 4. 图像归一化
        * 5. 图像标准化
        * 6. 图像反相
        * 7. 图像模糊
        * 8. 图像锐化
        * 9. 图像边缘检测
        * 10. 图像轮廓检测
        * 11. 图像分割
        * 12. 图像配准
        * 13. 图像增强
        * 14. 图像修复
        * 15. 图像增强
        */
        public:
        ImagePreprocess() = default;
        ~ImagePreprocess() = default;
        //!
        public:
        // 类型转换
        ImgProc::Img_Gray RGBToGray(const ImgProc::Img_RGB& rgb){
            ImgProc::Img_Gray gray(rgb.size(), std::vector<unsigned char>(rgb[0].size(), 0));
            for (int i = 0; i < rgb.size(); ++i) {
                for (int j = 0; j < rgb[0].size(); ++j) {
                    gray[i][j] = (rgb[i][j][0] + rgb[i][j][1] + rgb[i][j][2]) / 3;
                }
            }
            return gray;
        }
        ImgProc::Img_Gray RGBAToGray(const ImgProc::Img_RGBA& rgba){
            ImgProc::Img_Gray gray(rgba.size(), std::vector<unsigned char>(rgba[0].size(), 0));
            for (int i = 0; i < rgba.size(); ++i) {
                for (int j = 0; j < rgba[0].size(); ++j) {
                    gray[i][j] = rgba[i][j][0];
                }
            }
            return gray;
        }
        ImgProc::Img_Gray RGBFloatToGray(const ImgProc::Img_RGBFloat& rgb){
            ImgProc::Img_Gray gray(rgb.size(), std::vector<unsigned char>(rgb[0].size(), 0));
            for (int i = 0; i < rgb.size(); ++i) {
                for (int j = 0; j < rgb[0].size(); ++j) {
                    gray[i][j] = (rgb[i][j][0] + rgb[i][j][1] + rgb[i][j][2]) / 3;
                }
            }
            return gray;
        }

        //!
        public:
        // 单通道图像调整大小-最近邻插值
        template<typename PixelType>
        std::vector<std::vector<PixelType>> ResizeImageNearestNeighborInterpolation(const std::vector<std::vector<PixelType>>& img, int width, int height);
        
        // 多通道图像调整大小-最近邻插值
        template<typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> ResizeImageNearestNeighborInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, int width, int height);

        // 单通道图像调整大小-双线性插值
        template<typename PixelType>
        std::vector<std::vector<PixelType>> ResizeImageBilinearInterpolation(const std::vector<std::vector<PixelType>>& img, int width, int height);
        
        // 多通道图像调整大小-双线性插值
        template<typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> ResizeImageBilinearInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, int width, int height);
        
        // 单通道图像调整图像大小，按比例缩放-最近邻插值
        template<typename PixelType>
        std::vector<std::vector<PixelType>> ResizeWithRatioNearestNeighborInterpolation(const std::vector<std::vector<PixelType>>& img, float ratio);
        
        // 多通道图像调整图像大小，按比例缩放-最近邻插值
        template<typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> ResizeWithRatioNearestNeighborInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, float ratio);

        // 单通道图像调整图像大小，按比例缩放-双线性插值
        template<typename PixelType>
        std::vector<std::vector<PixelType>> ResizeWithRatioBilinearInterpolation(const std::vector<std::vector<PixelType>>& img, float ratio);
        
        // 多通道图像调整图像大小，按比例缩放-双线性插值
        template<typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> ResizeWithRatioBilinearInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, float ratio);

        //!
        public:
        // 图像灰度化
        template <typename PixelType>
        ImgProc::Img_Gray ConvertToGray(const std::vector<std::vector<std::vector<PixelType>>>& img);
        
        //!
        public:
        // 通用阈值应用（支持单通道和多通道）
        template <typename ImgType>
        ImgProc::Img_Binary ApplyThreshold(const ImgType& img, unsigned char threshold);

        // 灰度图像二值化（自适应阈值 - Otsu 算法）
        ImgProc::Img_Binary OtsuBinarizeImage(const ImgProc::Img_Gray& img);

        //!
        public:
        // 单通道图像归一化
        template <typename PixelType>
        std::vector<std::vector<float>> NormalizeImage(const std::vector<std::vector<PixelType>>& img, float min_val=0.0f, float max_val=1.0f);

        // 多通道图像归一化
        template <typename PixelType>
        std::vector<std::vector<std::vector<float>>> NormalizeImage(const std::vector<std::vector<std::vector<PixelType>>>& img, float min_val=0.0f, float max_val=1.0f);

        //!
        public:
        // 单通道图像标准化
        template <typename PixelType>
        std::vector<std::vector<float>> StandardizeImage(const std::vector<std::vector<PixelType>>& img);

        // 多通道图像标准化
        template <typename PixelType>
        std::vector<std::vector<std::vector<float>>> StandardizeImage(const std::vector<std::vector<std::vector<PixelType>>>& img);

        //!
        public:
        // 单通道图像反相
        template <typename PixelType>
        std::vector<std::vector<PixelType>> InvertImage(const std::vector<std::vector<PixelType>>& img);

        // 多通道图像反相
        template <typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> InvertImage(const std::vector<std::vector<std::vector<PixelType>>>& img);

        //!
        public:
        // 高斯模糊
        template <typename PixelType>
        std::vector<std::vector<float>> GaussianBlur(const std::vector<std::vector<PixelType>>& img, int kernel_size, float sigma);

        // 单通道图像模糊
        template <typename PixelType>
        std::vector<std::vector<PixelType>> BlurImage(const std::vector<std::vector<PixelType>>& img, const std::vector<std::vector<float>>& kernel);

        // 多通道图像模糊
        template <typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> BlurImage(const std::vector<std::vector<std::vector<PixelType>>>& img, const std::vector<std::vector<float>>& kernel);

        //!
        public:
        // 边缘检测
        template <typename PixelType>
        void ComputeGradient(const std::vector<std::vector<PixelType>>& img, std::vector<std::vector<float>>& magnitude, std::vector<std::vector<float>>& direction);

        // 单通道图像边缘检测-Sobel算子
        template <typename PixelType>
        std::vector<std::vector<float>> EdgeDetectionSobel(const std::vector<std::vector<PixelType>>& img);

        // 多通道图像边缘检测-Sobel算子
        template <typename PixelType>
        std::vector<std::vector<std::vector<float>>> EdgeDetectionSobel(const std::vector<std::vector<std::vector<PixelType>>>& img);

        // 单通道图像边缘检测-Canny算子
        template <typename PixelType>
        std::vector<std::vector<float>> EdgeDetectionCanny(const std::vector<std::vector<PixelType>>& img, int kernel_size = 3, float sigma = 1.0f, float low_thresh = 50.0f, float high_thresh = 150.0f);

        // 多通道图像边缘检测-Canny算子
        template <typename PixelType>
        std::vector<std::vector<std::vector<float>>> EdgeDetectionCanny(const std::vector<std::vector<std::vector<PixelType>>>& img,  int kernel_size=3, float sigma=1.0f, float low_thresh=50.0f, float high_thresh=150.0f);

        // 单通道图像边缘检测-Laplacian算子
        template <typename PixelType>
        std::vector<std::vector<float>> EdgeDetectionLaplacian(const std::vector<std::vector<PixelType>>& img);

        // 多通道图像边缘检测-Laplacian算子
        template <typename PixelType>
        std::vector<std::vector<std::vector<float>>> EdgeDetectionLaplacian(const std::vector<std::vector<std::vector<PixelType>>>& img);

        //!
        public:
        // 图像膨胀
        template <typename PixelType>
        std::vector<std::vector<PixelType>> Dilation(const std::vector<std::vector<PixelType>>& img, int kernel_size=3);
        
        // 图像腐蚀
        template <typename PixelType>
        std::vector<std::vector<PixelType>> Erosion(const std::vector<std::vector<PixelType>>& img, int kernel_size=3);
        //!
        // 非极大值抑制
        template <typename PixelType>
        std::vector<std::vector<float>> NonMaxSuppression(const std::vector<std::vector<PixelType>>& magnitude, const std::vector<std::vector<float>>& direction);
        // 双阈值
        template <typename PixelType>
        void DoubleThresholding(std::vector<std::vector<PixelType>>& edges, float low_thresh, float high_thresh);
        // 边缘连接（Hysteresis）
        template <typename PixelType>
        void EdgeTrackingByHysteresis(std::vector<std::vector<PixelType>>& edges);
    };

    //!
    template<typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::ResizeImageNearestNeighborInterpolation(const std::vector<std::vector<PixelType>>& img, int width, int height){
        int old_height = img.size();
        int old_width = img[0].size();
        std::vector<std::vector<PixelType>> result(height, std::vector<PixelType>(width));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int old_x=static_cast<int>(j*old_width/width);
                int old_y=static_cast<int>(i*old_height/height);
                result[i][j] = img[old_y][old_x];
            }
        }
        return result;
    }
        
    template<typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::ResizeImageNearestNeighborInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, int width, int height){
        int old_height = img.size();
        int old_width = img[0].size();
        int channels = img[0][0].size();

        ImgProc::Img_MultiChannel resized_img(height, std::vector<std::vector<PixelType>>(width, std::vector<PixelType>(channels)));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int old_x=static_cast<int>(j*old_width/width);
                int old_y=static_cast<int>(i*old_height/height);
                for (int c = 0; c < channels; ++c) {
                    resized_img[i][j][c] = img[old_y][old_x][c];
                }
            }
        }
        return resized_img;
    }

    template<typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::ResizeImageBilinearInterpolation(const std::vector<std::vector<PixelType>>& img, int width, int height){
        int old_width = img[0].size();
        int old_height = img.size();

        std::vector<std::vector<unsigned char>> resized_img(height, std::vector<unsigned char>(width));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float x = (float)(j * old_width) / width;
                float y = (float)(i * old_height) / height;
            
                int x1 = static_cast<int>(x);
                int y1 = static_cast<int>(y);
                int x2 = std::min(x1 + 1, old_width - 1);
                int y2 = std::min(y1 + 1, old_height - 1);
            
                float dx = x - x1;
                float dy = y - y1;
            
                unsigned char f11 = img[y1][x1];
                unsigned char f12 = img[y1][x2];
                unsigned char f21 = img[y2][x1];
                unsigned char f22 = img[y2][x2];
            
                unsigned char value = static_cast<unsigned char>(
                    f11 * (1 - dx) * (1 - dy) +
                    f12 * dx * (1 - dy) +
                    f21 * (1 - dx) * dy +
                    f22 * dx * dy
                );

                resized_img[i][j] = value;
            }
        }

        return resized_img;
    }
        
    template<typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::ResizeImageBilinearInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, int width, int height){
        int old_width = img[0].size();
        int old_height = img.size();
        int channels = img[0][0].size();

        std::vector<std::vector<std::vector<unsigned char>>> resized_img(height, 
            std::vector<std::vector<unsigned char>>(width, std::vector<unsigned char>(channels)));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float x = (float)(j * old_width) / width;
                float y = (float)(i * old_height) / height;

                int x1 = static_cast<int>(x);
                int y1 = static_cast<int>(y);
                int x2 = std::min(x1 + 1, old_width - 1);
                int y2 = std::min(y1 + 1, old_height - 1);

                float dx = x - x1;
                float dy = y - y1;

                for (int c = 0; c < channels; ++c) {
                    unsigned char f11 = img[y1][x1][c];
                    unsigned char f12 = img[y1][x2][c];
                    unsigned char f21 = img[y2][x1][c];
                    unsigned char f22 = img[y2][x2][c];

                    unsigned char value = static_cast<unsigned char>(
                        f11 * (1 - dx) * (1 - dy) +
                        f12 * dx * (1 - dy) +
                        f21 * (1 - dx) * dy +
                        f22 * dx * dy
                    );

                    resized_img[i][j][c] = value;
                }
            }
        }

        return resized_img;
    }

    template<typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::ResizeWithRatioNearestNeighborInterpolation(const std::vector<std::vector<PixelType>>& img, float ratio){
        if(ratio<=0.0f){
            throw std::invalid_argument("Ratio must be greater than 0.");
        }
        int old_height = img.size();
        int old_width = img[0].size();
        int new_height = static_cast<int>(old_height * ratio);
        int new_width = static_cast<int>(old_width * ratio);
        
        std::vector<std::vector<PixelType>> result(new_height, std::vector<PixelType>(new_width));
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                float old_x = (float)j / ratio;
                float old_y = (float)i / ratio;
                
                int x1 = static_cast<int>(old_x);
                int y1 = static_cast<int>(old_y);
                x1=std::min(x1, old_width-1);
                y1=std::min(y1, old_height-1);

                result[i][j] = img[y1][x1];
            }
        }
        return result;
    }
        
    template<typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::ResizeWithRatioNearestNeighborInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, float ratio){
        if(ratio<=0.0f){
            throw std::invalid_argument("Ratio must be greater than 0.");
        }
        int old_height = img.size();
        int old_width = img[0].size();

        int new_height = static_cast<int>(old_height * ratio);
        int new_width = static_cast<int>(old_width * ratio);

        std::vector<std::vector<PixelType>> resized_img(new_height, std::vector<PixelType>(new_width, 0));

        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                float old_x = j / ratio;
                float old_y = i / ratio;

                int x1 = static_cast<int>(old_x);
                int y1 = static_cast<int>(old_y);

                x1 = std::min(x1, old_width - 1);
                y1 = std::min(y1, old_height - 1);

                resized_img[i][j] = img[y1][x1];
            }
        }

        return resized_img;
    }

    template<typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::ResizeWithRatioBilinearInterpolation(const std::vector<std::vector<PixelType>>& img, float ratio){
        if (ratio <= 0.0f) {
            throw std::invalid_argument("Scale must be greater than 0.");
        }

        int old_height = img.size();
        int old_width = img[0].size();

        int new_height = static_cast<int>(old_height * ratio);
        int new_width = static_cast<int>(old_width * ratio);

        std::vector<std::vector<PixelType>> resized_img(new_height, std::vector<PixelType>(new_width, 0));
        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                float old_x = j / ratio;
                float old_y = i / ratio;

                int x1 = static_cast<int>(old_x);
                int y1 = static_cast<int>(old_y);

                float dx = old_x - x1;
                float dy = old_y - y1;

                int x2 = std::min(x1 + 1, old_width - 1);
                int y2 = std::min(y1 + 1, old_height - 1);

                PixelType top_left = img[y1][x1];
                PixelType top_right = img[y1][x2];
                PixelType bottom_left = img[y2][x1];
                PixelType bottom_right = img[y2][x2];

                PixelType top = top_left + dx * (top_right - top_left);
                PixelType bottom = bottom_left + dx * (bottom_right - bottom_left);
                resized_img[i][j] = top + dy * (bottom - top);
            }
        }
        return resized_img;
    }
        
    template<typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::ResizeWithRatioBilinearInterpolation(const std::vector<std::vector<std::vector<PixelType>>>& img, float ratio){
        if (ratio <= 0.0f) {
            throw std::invalid_argument("Scale must be greater than 0.");
        }

        int old_height = img.size();
        int old_width = img[0].size();
        int channels = img[0][0].size();

        int new_height = static_cast<int>(old_height * ratio);
        int new_width = static_cast<int>(old_width * ratio);

        std::vector<std::vector<std::vector<PixelType>>> resized_img(
            new_height, std::vector<std::vector<PixelType>>(new_width, std::vector<PixelType>(channels, 0)));

        for (int i = 0; i < new_height; ++i) {
            for (int j = 0; j < new_width; ++j) {
                float old_x = j / ratio;
                float old_y = i / ratio;

                int x1 = static_cast<int>(old_x);
                int y1 = static_cast<int>(old_y);

                float dx = old_x - x1;
                float dy = old_y - y1;

                int x2 = std::min(x1 + 1, old_width - 1);
                int y2 = std::min(y1 + 1, old_height - 1);

                for (int c = 0; c < channels; ++c) {
                    PixelType top_left = img[y1][x1][c];
                    PixelType top_right = img[y1][x2][c];
                    PixelType bottom_left = img[y2][x1][c];
                    PixelType bottom_right = img[y2][x2][c];

                    PixelType top = top_left + dx * (top_right - top_left);
                    PixelType bottom = bottom_left + dx * (bottom_right - bottom_left);
                    resized_img[i][j][c] = top + dy * (bottom - top);
                }
            }
        }

        return resized_img;
    }

    //!
    template <typename PixelType>
    ImgProc::Img_Gray ImagePreprocess::ConvertToGray(const std::vector<std::vector<std::vector<PixelType>>>& img){
        int height = img.size();
        int width = img[0].size();
        ImgProc::Img_Gray grayscaleImg(height, std::vector<unsigned char>(width));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                unsigned char r = img[i][j][0];
                unsigned char g = img[i][j][1];
                unsigned char b = img[i][j][2];
                unsigned char gray = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
                grayscaleImg[i][j] = gray;
            }
        }
        return grayscaleImg;
    }

    template <typename ImgType>
    ImgProc::Img_Binary ImagePreprocess::ApplyThreshold(const ImgType& img, unsigned char threshold){
        ImgProc::Img_Binary binaryImg(img.size(), std::vector<bool>(img[0].size(), false));
        for (int i = 0; i < img.size(); ++i) {
            for (int j = 0; j < img[0].size(); ++j) {
                binaryImg[i][j] = img[i][j] >= threshold;
            }
        }
        return binaryImg;
    }

    ImgProc::Img_Binary ImagePreprocess::OtsuBinarizeImage(const ImgProc::Img_Gray& img){
        int height = img.size();
        int width = img[0].size();

        std::vector<int> histogram(256, 0);
        int total_pixels = height * width;

        for (const auto& row : img) {
            for (const auto& pixel : row) {
                histogram[pixel]++;
            }
        }

        float total_gray_sum = 0;
        for (int i = 0; i < 256; ++i) {
            total_gray_sum += i * histogram[i];
        }

        float max_variance = 0.0f; // 最大类间方差
        int optimal_threshold = 0; // 最优阈值
        float background_sum = 0; // 背景的灰度值总和
        int background_pixels = 0; // 背景像素数

        for (int t = 0; t < 256; ++t) {
            background_pixels += histogram[t];
            if (background_pixels == 0) continue;

            int foreground_pixels = total_pixels - background_pixels;
            if (foreground_pixels == 0) break;

            background_sum += t * histogram[t];
            float background_mean = background_sum / background_pixels;
            float foreground_mean = (total_gray_sum - background_sum) / foreground_pixels;

            float between_variance = (float)background_pixels * foreground_pixels *
                (background_mean - foreground_mean) * (background_mean - foreground_mean);

            if (between_variance > max_variance) {
                max_variance = between_variance;
                optimal_threshold = t;
            }
        }

        ImgProc::Img_Binary binary_img(height, std::vector<unsigned char>(width, 0));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                binary_img[i][j] = (img[i][j] > optimal_threshold) ? 255 : 0;
            }
        }

        return binary_img;
    }

    //!
    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::NormalizeImage(const std::vector<std::vector<PixelType>>& img, float min_val, float max_val){
        if (img.empty() || img[0].empty()) {
            throw std::invalid_argument("Input image cannot be empty.");
        }

        PixelType img_min = img[0][0];
        PixelType img_max = img[0][0];
        for (const auto& row : img) {
            for (const auto& pixel : row) {
                img_min = std::min(img_min, pixel);
                img_max = std::max(img_max, pixel);
            }
        }

        if (img_max == img_min) {
            throw std::runtime_error("All pixels have the same value; cannot normalize.");
        }

        std::vector<std::vector<float>> normalized_img(img.size(), std::vector<float>(img[0].size(), 0.0f));
        for (size_t i = 0; i < img.size(); ++i) {
            for (size_t j = 0; j < img[i].size(); ++j) {
                normalized_img[i][j] = min_val + (img[i][j] - img_min) * (max_val - min_val) / (img_max - img_min);
            }
        }

        return normalized_img;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<float>>> ImagePreprocess::NormalizeImage(const std::vector<std::vector<std::vector<PixelType>>>& img, float min_val, float max_val){
        if (img.empty() || img[0].empty() || img[0][0].empty()) {
            throw std::invalid_argument("Input image cannot be empty.");
        }

        size_t channels = img[0][0].size();
        std::vector<PixelType> img_min(channels, std::numeric_limits<PixelType>::max());
        std::vector<PixelType> img_max(channels, std::numeric_limits<PixelType>::lowest());

        for (const auto& row : img) {
            for (const auto& pixel : row) {
                for (size_t c = 0; c < channels; ++c) {
                    img_min[c] = std::min(img_min[c], pixel[c]);
                    img_max[c] = std::max(img_max[c], pixel[c]);
                }
            }
        }

        for (size_t c = 0; c < channels; ++c) {
            if (img_max[c] == img_min[c]) {
                throw std::runtime_error("All pixels in one channel have the same value; cannot normalize.");
            }
        }

        std::vector<std::vector<std::vector<float>>> normalized_img(img.size(), std::vector<std::vector<float>>(img[0].size(), std::vector<float>(channels, 0.0f)));
        for (size_t i = 0; i < img.size(); ++i) {
            for (size_t j = 0; j < img[i].size(); ++j) {
                for (size_t c = 0; c < channels; ++c) {
                    normalized_img[i][j][c] = min_val + (img[i][j][c] - img_min[c]) * (max_val - min_val) / (img_max[c] - img_min[c]);
                }
            }
        }

        return normalized_img;
    }

    //!

    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::StandardizeImage(const std::vector<std::vector<PixelType>>& img){
        int height = img.size();
        int width = img[0].size();

        double sum = 0.0;
        for (const auto& row : img) {
            for (const auto& pixel : row) {
                sum += static_cast<double>(pixel);
            }
        }
        double mean = sum / (height * width);

        double variance = 0.0;
        for (const auto& row : img) {
            for (const auto& pixel : row) {
                double diff = static_cast<double>(pixel) - mean;
                variance += diff * diff;
            }
        }
        double stddev = std::sqrt(variance / (height * width));

        std::vector<std::vector<float>> standardized_img(height, std::vector<float>(width));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                standardized_img[i][j] = (static_cast<double>(img[i][j]) - mean) / stddev;
            }
        }

        return standardized_img;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<float>>> ImagePreprocess::StandardizeImage(const std::vector<std::vector<std::vector<PixelType>>>& img){
        int height = img.size();
        int width = img[0].size();
        int channels = img[0][0].size();

        std::vector<double> mean(channels, 0.0);
        std::vector<double> stddev(channels, 0.0);

        for (int c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    sum += static_cast<double>(img[i][j][c]);
                }
            }
            mean[c] = sum / (height * width);
        }

        for (int c = 0; c < channels; ++c) {
            double variance = 0.0;
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    double diff = static_cast<double>(img[i][j][c]) - mean[c];
                    variance += diff * diff;
                }
            }
            stddev[c] = std::sqrt(variance / (height * width));
        }

        std::vector<std::vector<std::vector<float>>> standardized_img(height, std::vector<std::vector<float>>(width, std::vector<float>(channels)));
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int c = 0; c < channels; ++c) {
                    standardized_img[i][j][c] = (static_cast<double>(img[i][j][c]) - mean[c]) / stddev[c];
                }
            }
        }

    return standardized_img;
    }
    //!

    template <typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::InvertImage(const std::vector<std::vector<PixelType>>& img){
        int height = img.size();
        int width = img[0].size();

        std::vector<std::vector<PixelType>> inverted_img(height, std::vector<PixelType>(width));

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                inverted_img[i][j] = 255 - img[i][j];
            }
        }
        return inverted_img;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::InvertImage(const std::vector<std::vector<std::vector<PixelType>>>& img){
        int height = img.size();
        int width = img[0].size();
        int channels = img[0][0].size();

        std::vector<std::vector<std::vector<PixelType>>> inverted_img(height, std::vector<std::vector<PixelType>>(width, std::vector<PixelType>(channels)));

        for(int i = 0; i < height; ++i){
            for(int j = 0; j < width; ++j){
                for(int c = 0; c < channels; ++c){
                    inverted_img[i][j][c] = 255 - img[i][j][c];
                }
            }
        }
        return inverted_img;
    }

    //!
    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::GaussianBlur(const std::vector<std::vector<PixelType>>& img, int kernel_size, float sigma){
        int height = img.size();
        int width = img[0].size();
        ImgProc::SingleChannelImage blurred(height, std::vector<float>(width, 0));

        int half_size = kernel_size / 2;
        std::vector<std::vector<float>> kernel(kernel_size, std::vector<float>(kernel_size, 0));

        float sum = 0.0;
        for (int i = -half_size; i <= half_size; ++i) {
            for (int j = -half_size; j <= half_size; ++j) {
                float value = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
                kernel[i + half_size][j + half_size] = value;
                sum += value;
            }
        }

        for (auto& row : kernel)
            for (auto& val : row)
                val /= sum;

        for (int i = half_size; i < height - half_size; ++i) {
            for (int j = half_size; j < width - half_size; ++j) {
                float blurred_value = 0;
                for (int ki = -half_size; ki <= half_size; ++ki) {
                    for (int kj = -half_size; kj <= half_size; ++kj) {
                        blurred_value += img[i + ki][j + kj] * kernel[ki + half_size][kj + half_size];
                    }
                }
                blurred[i][j] = blurred_value;
            }
        }
        return blurred;
    }

    template <typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::BlurImage(const std::vector<std::vector<PixelType>>& img, const std::vector<std::vector<float>>& kernel){
         int kernel_size = kernel.size();
        if (kernel_size % 2 == 0 || kernel_size != kernel[0].size()) {
            throw std::invalid_argument("Kernel must be square and have an odd size.");
        }

        int height = img.size();
        int width = img[0].size();
        int half_k = kernel_size / 2;

        std::vector<std::vector<PixelType>> result(height, std::vector<PixelType>(width, 0));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float sum = 0.0f;
                for (int ki = -half_k; ki <= half_k; ++ki) {
                    for (int kj = -half_k; kj <= half_k; ++kj) {
                        int ni = i + ki;
                        int nj = j + kj;
                        if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                            sum += img[ni][nj] * kernel[ki + half_k][kj + half_k];
                        }
                    }
                }
                result[i][j] = static_cast<PixelType>(std::round(sum));
            }
        }

        return result;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<PixelType>>> ImagePreprocess::BlurImage(const std::vector<std::vector<std::vector<PixelType>>>& img, const std::vector<std::vector<float>>& kernel){
         int kernel_size = kernel.size();
        if (kernel_size % 2 == 0 || kernel_size != kernel[0].size()) {
            throw std::invalid_argument("Kernel must be square and have an odd size.");
        }

        int height = img.size();
        int width = img[0].size();
        int channels = img[0][0].size();
        int half_k = kernel_size / 2;

        std::vector<std::vector<std::vector<PixelType>>> result(height, std::vector<std::vector<PixelType>>(width, std::vector<PixelType>(channels, 0)));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int c = 0; c < channels; ++c) {
                    float sum = 0.0f;
                    for (int ki = -half_k; ki <= half_k; ++ki) {
                        for (int kj = -half_k; kj <= half_k; ++kj) {
                            int ni = i + ki;
                            int nj = j + kj;
                            if (ni >= 0 && ni < height && nj >= 0 && nj < width) {
                                sum += img[ni][nj][c] * kernel[ki + half_k][kj + half_k];
                            }
                        }
                    }
                    result[i][j][c] = static_cast<PixelType>(std::round(sum));
                }
            }
        }

        return result;
    }

    //!

    template <typename PixelType>
    void ImagePreprocess::ComputeGradient(const std::vector<std::vector<PixelType>>& img, std::vector<std::vector<float>>& magnitude, std::vector<std::vector<float>>& direction){
        int height = img.size();
        int width = img[0].size();

        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                float grad_x = 0, grad_y = 0;

                for (int ki = -1; ki <= 1; ++ki) {
                    for (int kj = -1; kj <= 1; ++kj) {
                        grad_x += img[i + ki][j + kj] * ImgProc::Sobel::Gx[ki + 1][kj + 1];
                        grad_y += img[i + ki][j + kj] * ImgProc::Sobel::Gy[ki + 1][kj + 1];
                    }
                }

                magnitude[i][j] = std::sqrt(grad_x * grad_x + grad_y * grad_y);
                direction[i][j] = std::atan2(grad_y, grad_x) * 180.0 / M_PI;
            }
        }
    }

    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::EdgeDetectionSobel(const std::vector<std::vector<PixelType>>& img){
        int height = img.size();
        int width = img[0].size();

        std::vector<std::vector<float>> edges(height, std::vector<float>(width, 0));

        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                float gradient_x = 0.0f;
                float gradient_y = 0.0f;

                for (int ki = -1; ki <= 1; ++ki) {
                    for (int kj = -1; kj <= 1; ++kj) {
                        gradient_x += img[i + ki][j + kj] * ImgProc::Sobel::Gx[ki + 1][kj + 1];
                        gradient_y += img[i + ki][j + kj] * ImgProc::Sobel::Gy[ki + 1][kj + 1];
                    }
                }

                edges[i][j] = std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);
            }
        }

        return edges;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<float>>> ImagePreprocess::EdgeDetectionSobel(const std::vector<std::vector<std::vector<PixelType>>>& img){
        int height = img.size();
        int width = img[0].size();
        int channels = img[0][0].size();

        std::vector<std::vector<std::vector<float>>> edges(height, std::vector<std::vector<float>>(width, std::vector<float>(channels, 0)));

        for (int c = 0; c < channels; ++c) {
            std::vector<std::vector<PixelType>> single_channel(height, std::vector<PixelType>(width, 0));
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    single_channel[i][j] = img[i][j][c];
                }
            }
            
            std::vector<std::vector<float>> single_channel_edges = EdgeDetectionSobel(single_channel);

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    edges[i][j][c] = single_channel_edges[i][j];
                }
            }
        }

        return edges;
    }

    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::EdgeDetectionCanny(const std::vector<std::vector<PixelType>>& img, int kernel_size, float sigma, float low_thresh, float high_thresh){
        ImgProc::SingleChannelImage blurred = GaussianBlur(img, kernel_size, sigma);

        ImgProc::SingleChannelImage magnitude(img.size(), std::vector<float>(img[0].size(), 0));
        ImgProc::SingleChannelImage direction(img.size(), std::vector<float>(img[0].size(), 0));
        ComputeGradient(blurred, magnitude, direction);

        ImgProc::SingleChannelImage suppressed = NonMaxSuppression(magnitude, direction);

        DoubleThresholding(suppressed, low_thresh, high_thresh);

        EdgeTrackingByHysteresis(suppressed);

        return suppressed;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<float>>> ImagePreprocess::EdgeDetectionCanny(const std::vector<std::vector<std::vector<PixelType>>>& img,  int kernel_size, float sigma, float low_thresh, float high_thresh){
        int height = img.size();
        int width = img[0].size();

        ImgProc::MultiChannelImage result(height, std::vector<std::vector<float>>(width, std::vector<float>(3, 0)));

        for (int c = 0; c < 3; ++c) {
            ImgProc::SingleChannelImage channel(height, std::vector<float>(width, 0));

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    channel[i][j] = img[i][j][c];
                }
            }

            ImgProc::SingleChannelImage edges = EdgeDetectionCanny(channel, kernel_size, sigma, low_thresh, high_thresh);

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    result[i][j][c] = edges[i][j];
                }
            }
        }

        return result;
    }

    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::EdgeDetectionLaplacian(const std::vector<std::vector<PixelType>>& img){
        int height = img.size();
        int width = img[0].size();
        ImgProc::SingleChannelImage result(height, std::vector<float>(width, 0));

        int kernel_size = 3;
        int half_size = kernel_size / 2;

        for (int i = half_size; i < height - half_size; ++i) {
            for (int j = half_size; j < width - half_size; ++j) {
                float value = 0.0f;

                for (int ki = -half_size; ki <= half_size; ++ki) {
                    for (int kj = -half_size; kj <= half_size; ++kj) {
                        value += img[i + ki][j + kj] * ImgProc::Laplacian::LaplacianEdgeDetection[ki + half_size][kj + half_size];
                    }
                }

                result[i][j] = std::abs(value);
            }
        }

        return result;
    }

    template <typename PixelType>
    std::vector<std::vector<std::vector<float>>> ImagePreprocess::EdgeDetectionLaplacian(const std::vector<std::vector<std::vector<PixelType>>>& img){
        int height = img.size();
        int width = img[0].size();
        int channels = img[0][0].size();

        ImgProc::MultiChannelImage result(height, std::vector<std::vector<float>>(width, std::vector<float>(channels, 0)));

        for (int c = 0; c < channels; ++c) {
            ImgProc::SingleChannelImage channel(height, std::vector<float>(width, 0));
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    channel[i][j] = img[i][j][c];
                }
            }

            ImgProc::SingleChannelImage edges = EdgeDetectionLaplacian(channel);

            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    result[i][j][c] = edges[i][j];
                }
            }
        }

        return result;
    }

    //!
    template <typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::Dilation(const std::vector<std::vector<PixelType>>& img, int kernel_size){
        int height = img.size();
        int width = img[0].size();
        int offset = kernel_size / 2;

        std::vector<std::vector<PixelType>> result(height, std::vector<PixelType>(width, 0));
        for (int i = offset; i < height - offset; ++i) {
            for (int j = offset; j < width - offset; ++j) {
                unsigned char max_value = 0;
                for (int ki = -offset; ki <= offset; ++ki) {
                    for (int kj = -offset; kj <= offset; ++kj) {
                        max_value = std::max(max_value, img[i + ki][j + kj]);
                    }
                }
                result[i][j] = max_value;
            }
        }

        return result;
    }
        
    template <typename PixelType>
    std::vector<std::vector<PixelType>> ImagePreprocess::Erosion(const std::vector<std::vector<PixelType>>& img, int kernel_size){
        int height = img.size();
        int width = img[0].size();
        std::vector<std::vector<PixelType>> result(height, std::vector<PixelType>(width, 0));

        int offset = kernel_size / 2;

        for (int i = offset; i < height - offset; ++i) {
            for (int j = offset; j < width - offset; ++j) {
                unsigned char min_value = 255;
                for (int ki = -offset; ki <= offset; ++ki) {
                    for (int kj = -offset; kj <= offset; ++kj) {
                        min_value = std::min(min_value, img[i + ki][j + kj]);
                    }
                }
                result[i][j] = min_value;
            }
        }

        return result;
    }
    
    //!

    template <typename PixelType>
    std::vector<std::vector<float>> ImagePreprocess::NonMaxSuppression(const std::vector<std::vector<PixelType>>& magnitude, const std::vector<std::vector<float>>& direction){
        int height = magnitude.size();
        int width = magnitude[0].size();
        std::vector<std::vector<float>> suppressed(height, std::vector<float>(width, 0));

        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                float angle = direction[i][j];
                angle = angle < 0 ? angle + 180 : angle;

                float q = 0, r = 0;

                if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                    q = magnitude[i][j + 1];
                    r = magnitude[i][j - 1];
                } else if (angle >= 22.5 && angle < 67.5) {
                    q = magnitude[i + 1][j - 1];
                    r = magnitude[i - 1][j + 1];
                } else if (angle >= 67.5 && angle < 112.5) {
                    q = magnitude[i + 1][j];
                    r = magnitude[i - 1][j];
                } else if (angle >= 112.5 && angle < 157.5) {
                    q = magnitude[i - 1][j - 1];
                    r = magnitude[i + 1][j + 1];
                }

                if (magnitude[i][j] >= q && magnitude[i][j] >= r)
                    suppressed[i][j] = magnitude[i][j];
                else
                    suppressed[i][j] = 0;
            }
        }

        return suppressed;
    }

    template <typename PixelType>
    void ImagePreprocess::DoubleThresholding(std::vector<std::vector<PixelType>>& edges, float low_thresh, float high_thresh){
        int height = edges.size();
        int width = edges[0].size();

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                if (edges[i][j] >= high_thresh) {
                    edges[i][j] = 255;
                } else if (edges[i][j] < low_thresh) {
                    edges[i][j] = 0;
                } else {
                    edges[i][j] = 128; // 弱边缘
                }
            }
        }
    }

    template <typename PixelType>
    void ImagePreprocess::EdgeTrackingByHysteresis(std::vector<std::vector<PixelType>>& edges) {
        int height = edges.size();
        int width = edges[0].size();

        for (int i = 1; i < height - 1; ++i) {
            for (int j = 1; j < width - 1; ++j) {
                if (edges[i][j] == 128) {
                    bool connected_to_strong_edge = false;
                    for (int di = -1; di <= 1; ++di) {
                        for (int dj = -1; dj <= 1; ++dj) {
                            if (edges[i + di][j + dj] == 255) {
                                connected_to_strong_edge = true;
                                break;
                            }
                        }
                        if (connected_to_strong_edge)
                            break;
                    }

                    if (connected_to_strong_edge)
                        edges[i][j] = 255;
                    else
                        edges[i][j] = 0;
                }
            }
        }
    }
}

#endif // IMAGEPREPROCESSING_HPP