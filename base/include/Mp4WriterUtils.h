class Mp4WriterSinkUtils
{
public:
    Mp4WriterSinkUtils();
    std::string getFilenameForNextFrame(std::string &basefolder, uint32_t chunkSize);
    ~Mp4WriterSinkUtils();
};
