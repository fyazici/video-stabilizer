/* standard includes */
#include <cmath>
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <iomanip>
#include <thread>
#include <atomic>
#include <numeric>

/* opencv includes */
#include <opencv2/opencv.hpp>

/* boost includes */
#include <boost/program_options.hpp>
#include <boost/range/combine.hpp>
#include <boost/filesystem.hpp>

/* glog includes */
#include <glog/logging.h>

struct AppConfiguration {
    std::string input_path, output_path, codec;
    size_t window_size;
    
    int get_codec_index() const {
        if (codec.size() == 4)
            return CV_FOURCC(codec[0], codec[1], codec[2], codec[3]);
        else
            return -1;
    }
};

std::pair<bool, AppConfiguration> parse_program_options(int argc, char **argv) {
    namespace po = boost::program_options;
    
    AppConfiguration cfg;
    
    try 
    {
        po::options_description desc{"Options"};
        desc.add_options()
            ("help,h", "Display this help message")
            ("input-file,i", po::value<std::string>(&cfg.input_path), "Input filename")
            ("output-file,o", po::value<std::string>(&cfg.output_path), "Output filename")
            ("codec,c", po::value<std::string>(&cfg.codec)->default_value("MP4V"), "Codec fourcc code")
            ("frames,n", po::value<size_t>(&cfg.window_size)->default_value(30), "Smoothing window size for trajectory stabilization");
            
        po::command_line_parser parser{argc, argv};
        parser.options(desc).allow_unregistered().style(
            po::command_line_style::default_style | 
            po::command_line_style::allow_slash_for_short);
        po::parsed_options parsed_options = parser.run();
        
        po::variables_map vm;
        po::store(parsed_options, vm);
        po::notify(vm);
        
        if (vm.count("help")) {
            std::cout << desc;
            return std::make_pair(false, AppConfiguration{});
        } else if (!vm.count("input-file") || !vm.count("output-file")) {
            std::cout << "Input and output filenames are required.\n" << desc;
            return std::make_pair(false, AppConfiguration{});
        }
        
    } 
    catch(const po::error &ex) 
    {
        std::cerr << ex.what() << std::endl;
        exit(-1);
    }
    
    return std::make_pair(true, cfg);
}

// adapted from https://stackoverflow.com/a/21995693/4911614
/**
 * @private
 */
template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ... Args>
    static typename TimeT::rep execution(F func, Args&&... args)
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        func(std::forward<Args>(args)...);
        
        auto duration = std::chrono::duration_cast<TimeT>(
            std::chrono::high_resolution_clock::now() - start);
        
        return duration.count();
    }
};

struct StabilizerParams {
    int feature_maxCorners;
    double feature_qualityLevel;
    double feature_minDistance;
    size_t smooth_windowSize;
    
    StabilizerParams() :
        feature_maxCorners{200}, 
        feature_qualityLevel{0.01}, 
        feature_minDistance{30.}, 
        smooth_windowSize{}
    {}
};

/**
 * @class RigidMotion2D
 * Represents a (usually delta) rigid motion by rotation and translation.
 * Transformation order is assumed to be first rotation and then translation.
 */
struct RigidMotion2D {
    double tx; /*!< Translation in the \f$ x \f$ axis */
    double ty; /*!< Translation in the \f$ y \f$ axis */
    double angle; /*!< Rotation amount \f$ \theta \f$ */
    
    RigidMotion2D () : tx(0.), ty(0.), angle(0.) {}
    
    RigidMotion2D (double tx_, double ty_, double angle_) : tx (tx_), ty (ty_), angle(angle_) {}
    
    /** 
     * @brief Initialize a RigidMotion2D from a rigid transformation matrix
     * 
     * Transformation matrix must be in the form of a rigid transform:
     * \f$\begin{bmatrix}
     * \cos(\theta) & -\sin(\theta) & t_x \\
     * \sin(\theta) & \cos(\theta) & t_y \end{bmatrix}\f$
     */
    RigidMotion2D(cv::Mat_<double> rigid_) {
        tx = rigid_(0, 2);
        ty = rigid_(1, 2);
        angle = std::atan2(rigid_(1, 0), rigid_(0, 0));
    }
    
    RigidMotion2D operator+(const RigidMotion2D& other) const {
        return {tx+other.tx, ty+other.ty, angle+other.angle};
    }
    
    RigidMotion2D operator-(const RigidMotion2D& other) const {
        return {tx-other.tx, ty-other.ty, angle-other.angle};
    }
    
    RigidMotion2D operator*(double k) const {
        return {tx*k, ty*k, angle*k};
    }
    
    RigidMotion2D operator/(double k) const {
        return {tx/k, ty/k, angle/k};
    }
    
    RigidMotion2D operator-() const {
        return {-tx, -ty, -angle};
    }
    
    RigidMotion2D& operator+=(const RigidMotion2D& rhs) {
        tx += rhs.tx;
        ty += rhs.ty;
        angle += rhs.angle;
        return *this;
    }
    
    RigidMotion2D& operator-=(const RigidMotion2D& rhs) {
        tx -= rhs.tx;
        ty -= rhs.ty;
        angle -= rhs.angle;
        return *this;
    }
    
    RigidMotion2D& operator*=(double k) {
        tx *= k;
        ty *= k;
        angle *= k;
        return *this;
    }
    
    RigidMotion2D& operator/=(double k) {
        tx /= k;
        ty /= k;
        angle /= k;
        return *this;
    }
    
    cv::Mat_<double> rigid() const {
        cv::Mat_<double> rigid(2, 3);
        rigid(0, 0) =  std::cos(angle);
        rigid(0, 1) = -std::sin(angle);
        rigid(1, 0) =  std::sin(angle);
        rigid(1, 1) =  std::cos(angle);
        rigid(0, 2) =  tx;
        rigid(1, 2) =  ty;
        return rigid;
    }
    
    double norm() const {
        return std::sqrt(tx*tx + ty*ty);
    }
    
    friend std::ostream& operator<<(std::ostream& os, const RigidMotion2D& tr) {
        os << "[" << tr.tx << "," << tr.ty << "," << tr.angle << "]";
        return os;
    }
};

/**
 * @class Trajectory
 * An ordered list of RigidMotion2D pieces. Implemented as a `std::vector<RigidMotion2D>`
 */
using Trajectory = std::vector<RigidMotion2D>;

struct TransformState {
    StabilizerParams params;
    Trajectory d_trajectory;
};

static volatile std::atomic<int> frameNumber{0};

void obtainDTrajectory(cv::VideoCapture& cap, TransformState& transformState, int numframes) {
    using vp2f = std::vector<cv::Point2f>;
    
    cv::Mat previous_frame, current_frame;
    cv::Mat previous_gray, current_gray;
    
    if (numframes == -1)
        numframes = int(cap.get(CV_CAP_PROP_FRAME_COUNT));
    
    cap >> previous_frame;
    cv::cvtColor(previous_frame, previous_gray, cv::COLOR_BGR2GRAY);
    
    for (int i = 0; i < numframes; ++i, current_gray.copyTo(previous_gray))
    {
        ++frameNumber;
        
        vp2f previous_points, current_points;
        vp2f good_previous_points, good_current_points;
        std::vector<unsigned char> status;
        std::vector<float> error;
        
        cap >> current_frame;
        if (current_frame.data == nullptr)
            break;
        
        cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
        
        cv::goodFeaturesToTrack(previous_gray, previous_points, 
                                transformState.params.feature_maxCorners, transformState.params.feature_qualityLevel, transformState.params.feature_minDistance);
        
        LOG(INFO) << "Found trackable features: " << previous_points.size();
        if (previous_points.size() < 4) {
            LOG(WARNING) << "Not enough trackable features found. Fallback latest valid transform.";
            transformState.d_trajectory.push_back(transformState.d_trajectory.back());
            continue;
        } 
        
        cv::calcOpticalFlowPyrLK(previous_gray, current_gray, previous_points, current_points, status, error);
        
        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                good_previous_points.emplace_back(previous_points[i]);
                good_current_points.emplace_back(current_points[i]);
            }
        }
        
        LOG(INFO) << "Succesfully tracked features: " << good_previous_points.size();
        if (good_previous_points.size() < 4) {
            LOG(WARNING) << "Not enough features had been succesfully tracked. Fallback latest valid transform.";
            transformState.d_trajectory.push_back(transformState.d_trajectory.back());
            continue;
        }
        
        cv::Mat rigid = cv::estimateRigidTransform(good_previous_points, good_current_points, false);
        
        if (rigid.data == nullptr) {
            LOG(WARNING) << "Could not find a valid rigid transform. Fallback latest valid transform.";
            transformState.d_trajectory.push_back(transformState.d_trajectory.back());
            continue;
        }

        transformState.d_trajectory.emplace_back(rigid);
    }
}

template<typename T>
std::string join(std::string&&, T t_)
{
    std::stringstream ss;
    ss << t_;
    return ss.str();
}

template<typename T, typename ... Args>
std::string join(std::string&& sep_, T t_, Args ... args_)
{
    std::stringstream ss;
    ss << t_ << sep_ << join(std::forward<std::string>(sep_), args_...);
    return ss.str();
}

void smoothDTrajectory(TransformState& transformState) {
    size_t count = transformState.d_trajectory.size();
    Trajectory trajectory(count), smooth_trajectory(count), d_smooth_trajectory(count);
    
    std::partial_sum(transformState.d_trajectory.begin(), transformState.d_trajectory.end(), trajectory.begin());
    
    for (size_t i = 0; i < count; ++i) {
        size_t win_beg = std::max<int>(i - transformState.params.smooth_windowSize, 0);
        size_t win_end = std::min<int>(i + transformState.params.smooth_windowSize + 1, trajectory.size()); // exclusive
        
        RigidMotion2D acc = std::accumulate(trajectory.begin() + win_beg, trajectory.begin() + win_end, RigidMotion2D {});
        acc /= (win_end - win_beg);
        smooth_trajectory[i] = acc;
        
        acc -= trajectory[i];
        d_smooth_trajectory[i] = acc;
    }
    
    {
        std::ofstream ofs("trajectory.csv");
        ofs << "index,trajectory_norm,trajectory_angle,smooth_norm,smooth_angle\n";
        for (size_t i = 0; i < count; ++i) {
            ofs << join(",", i, 
                        trajectory[i].norm(), trajectory[i].angle,
                        smooth_trajectory[i].norm(), smooth_trajectory[i].angle) << "\n";
        }
        ofs.close();
    }
    
    transformState.d_trajectory = d_smooth_trajectory;
}

void stabilizeVideo(cv::VideoCapture& src, cv::VideoWriter& dst, TransformState& transformState, int numframes = -1) {
    if (numframes == -1)
        numframes = src.get(CV_CAP_PROP_FRAME_COUNT);
    
    auto width = int(src.get(CV_CAP_PROP_FRAME_WIDTH));
    auto height = int(src.get(CV_CAP_PROP_FRAME_HEIGHT));
    
    cv::Mat inputFrame(height, width, CV_8UC3), outputFrame(height, width, CV_8UC3);
    
    for (int i = 0; i < numframes; ++i)
    {
        src >> inputFrame;
        if (inputFrame.data == nullptr)
            break;
        
        // LOG(INFO) << "Stabilizing frame idx: " << i;
        cv::warpAffine(inputFrame, outputFrame, transformState.d_trajectory[i].rigid(), inputFrame.size());
        
        dst << outputFrame;
    }
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]); /* no gflags support */
    bool args_parsed;
    AppConfiguration config;
    std::tie(args_parsed, config) = parse_program_options(argc, argv);
    
    if (!args_parsed)
        return -1;
    
    auto src = cv::VideoCapture(config.input_path);
    
    if (!src.isOpened()) {
        LOG(ERROR) << "Cannot open input video file.\n";
        exit(-1);
    }
    
    int width = int(src.get(CV_CAP_PROP_FRAME_WIDTH)),
        height = int(src.get(CV_CAP_PROP_FRAME_HEIGHT));
    double frameRate = src.get(CV_CAP_PROP_FPS);
    int frameCount = int(src.get(CV_CAP_PROP_FRAME_COUNT));
    std::cout << frameRate << std::endl;
    
    auto dst = cv::VideoWriter(config.output_path, config.get_codec_index(), frameRate, cv::Size{width, height}, true);
    
    if (!dst.isOpened()) {
        LOG(ERROR) << "Cannot create output video file.\n";
        exit(-1);
    }
    
    std::thread tProgress ([=]() {
        using namespace std::chrono_literals;
        using namespace std::chrono;
        
        std::cout << std::setprecision(1) << std::fixed;
        
        auto start = high_resolution_clock::now();
        do {
            auto elapsed = duration_cast< duration<double, std::ratio<1, 1>> >(high_resolution_clock::now() - start).count();
            std::cout << "Processed " << double(100 * ::frameNumber) / frameCount << "%." 
                << "ETA: " << (frameCount - ::frameNumber) / (::frameNumber / elapsed) << "s"
                << std::endl;
            std::this_thread::sleep_for(500ms);
        } while(::frameNumber < frameCount);
    });
    
    TransformState state;
    state.params.smooth_windowSize = config.window_size;
    
    {
        auto elapsed =  measure<>::execution(obtainDTrajectory, src, state, -1);
        LOG(INFO) << "obtainDTrajectory execution took: " << elapsed << "ms";
    }
    
    /* seek begin */
    src.set(CV_CAP_PROP_POS_FRAMES, 0);
    
    {
        auto elapsed =  measure<>::execution(smoothDTrajectory, state);
        LOG(INFO) << "smoothDTrajectory execution took: " << elapsed << "ms";
    }
    
    {
        auto elapsed =  measure<>::execution(stabilizeVideo, src, dst, state, -1);
        LOG(INFO) << "stabilizeVideo execution took: " << elapsed << "ms";
    }

    src.release();
    dst.release();
    
    tProgress.join();
    
    return 0;
}

