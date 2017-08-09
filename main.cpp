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
    
    int get_codec_index() const {
        if (codec.size() == 4)
            return CV_FOURCC(codec[0], codec[1], codec[2], codec[3]);
        else
            return -1;
    }
};

AppConfiguration parse_program_options(int argc, char **argv) {
    namespace po = boost::program_options;
    
    AppConfiguration cfg;
    
    try 
    {
        po::options_description desc{"Options"};
        desc.add_options()
            ("input-file,i", po::value<std::string>(&cfg.input_path)->required(), "Input filename")
            ("output-file,o", po::value<std::string>(&cfg.output_path)->required(), "Output filename")
            ("codec,c", po::value<std::string>(&cfg.codec)->default_value("MP4V"), "Codec fourcc code");
            
        po::command_line_parser parser{argc, argv};
        parser.options(desc).allow_unregistered().style(
            po::command_line_style::default_style | 
            po::command_line_style::allow_slash_for_short);
        po::parsed_options parsed_options = parser.run();
        
        po::variables_map vm;
        po::store(parsed_options, vm);
        po::notify(vm);
        
    } 
    catch(const po::error &ex) 
    {
        std::cerr << ex.what() << std::endl;
        exit(-1);
    }
    
    return cfg;
}

// adapted from https://stackoverflow.com/a/21995693/4911614
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
        smooth_windowSize{30}
    {}
};

struct Transrot {
    double tlx, tly, angle;
    
    Transrot() : tlx(0.), tly(0.), angle(0.) {}
    
    Transrot(double tlx_, double tly_, double angle_) : tlx(tlx_), tly(tly_), angle(angle_) {}
    
    Transrot operator+(const Transrot& other) const {
        return Transrot{tlx+other.tlx, tly+other.tly, angle+other.angle};
    }
    
    Transrot operator-(const Transrot& other) const {
        return Transrot{tlx-other.tlx, tly-other.tly, angle-other.angle};
    }
    
    Transrot operator*(double k) const {
        return Transrot{tlx*k, tly*k, angle*k};
    }
    
    Transrot operator/(double k) const {
        return Transrot{tlx/k, tly/k, angle/k};
    }
    
    cv::Mat get_rigid() const {
        cv::Mat rigid(2, 3, CV_64F);
        rigid.at<double>(0, 0) =  std::cos(angle);
        rigid.at<double>(0, 1) = -std::sin(angle);
        rigid.at<double>(1, 0) =  std::sin(angle);
        rigid.at<double>(1, 1) =  std::cos(angle);
        rigid.at<double>(0, 2) =           tlx   ;
        rigid.at<double>(1, 2) =           tly   ;
        return rigid;
    }
};

using Trajectory = std::vector<Transrot>;

std::ostream& operator<<(std::ostream& os, const Transrot& tr) {
    os << tr.tlx << "," << tr.tly << "," << tr.angle;
    return os;
}

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
        
        Transrot tr;
        tr.tlx = rigid.at<double>(0, 2);
        tr.tly = rigid.at<double>(1, 2);
        tr.angle = std::atan2(rigid.at<double>(1, 0), rigid.at<double>(0, 0));
        transformState.d_trajectory.emplace_back(tr);
    }
}

void smoothDTrajectory(TransformState& transformState) {
    Trajectory trajectory(transformState.d_trajectory.size());
    Trajectory smooth_trajectory(transformState.d_trajectory.size());
    Trajectory d_smooth_trajectory(transformState.d_trajectory.size());
    
    std::partial_sum(transformState.d_trajectory.begin(), transformState.d_trajectory.end(), trajectory.begin());
    
    for (size_t i = 1; i < trajectory.size(); ++i) {
        size_t win_beg = std::max<int>(i - transformState.params.smooth_windowSize, 0);
        size_t win_end = std::min<int>(i + transformState.params.smooth_windowSize + 1, trajectory.size()); // exclusive
        
        Transrot acc;
        for (auto k = win_beg; k < win_end; ++k)
            acc = acc + trajectory[k];
        acc = acc/(win_end - win_beg) - trajectory[i];
        d_smooth_trajectory[i] = acc;
    }
    
    if ( /* DISABLES CODE */ ( 0 ) )
    {
        std::ofstream ofsTrajectory{"trajectory.csv", std::ios::out};
        for (size_t i = 0; i < trajectory.size(); ++i) {
            ofsTrajectory 
                << transformState.d_trajectory[i] << ","
                << trajectory[i] << ","
                << smooth_trajectory[i] << ","
                << d_smooth_trajectory[i] << "\n";
        }
        ofsTrajectory.close();
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
        cv::warpAffine(inputFrame, outputFrame, transformState.d_trajectory[i].get_rigid(), inputFrame.size());
        
        dst << outputFrame;
    }
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]); /* no gflags support */
    auto config = parse_program_options(argc, argv);
    
    std::cout << config.input_path << config.output_path << config.codec << "\n";
    
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
            LOG(INFO) << "Processed " << double(100 * ::frameNumber) / frameCount << "%." 
                << "ETA: " << (frameCount - ::frameNumber) / (::frameNumber / elapsed) << "s"
                << std::endl;
            std::this_thread::sleep_for(500ms);
        } while(::frameNumber < frameCount);
        LOG(INFO) << "\nFinished.\n";
    });
    
    TransformState state;
    
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
