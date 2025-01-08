#ifndef IMAGEPREPROCESSING_HPP
#define IMAGEPREPROCESSING_HPP

#include <iostream>
#include <vector>

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
        */
        public:
        ImagePreprocess() = default;
        ~ImagePreprocess() = default;
        
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
        // 单通道图像模糊
        template <typename PixelType>
        std::vector<std::vector<PixelType>> BlurImage(const std::vector<std::vector<PixelType>>& img, const std::vector<std::vector<float>>& kernel);

        // 多通道图像模糊
        template <typename PixelType>
        std::vector<std::vector<std::vector<PixelType>>> BlurImage(const std::vector<std::vector<std::vector<PixelType>>>& img, const std::vector<std::vector<float>>& kernel);

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
}

#endif // IMAGEPREPROCESSING_HPP