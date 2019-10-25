//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <string>
#include <dirent.h>
#include <iostream>
#include <sys/types.h>
#include "../InferenceTest.hpp"
#include "../ImagePreprocessor.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        std::vector<ImageSet> imageSet = {};
        std::vector<unsigned int> testcases;

        DIR* directory = opendir( "./" );
        struct dirent * dp;
        unsigned int case_counter = 0;
        while( ( dp = readdir( directory ) ) != NULL ){
            unsigned int size = strlen( dp -> d_name );
            //cout << dp -> d_name << endl;
            if( size > 4 &&
                ( dp -> d_name[ size - 1 ] == 'g' || dp -> d_name[ size - 1 ] == 'G' ) &&
                ( dp -> d_name[ size - 2 ] == 'e' || dp -> d_name[ size - 2 ] == 'E' ) &&
                ( dp -> d_name[ size - 3 ] == 'p' || dp -> d_name[ size - 3 ] == 'P' ) &&
                ( dp -> d_name[ size - 4 ] == 'J' || dp -> d_name[ size - 4 ] == 'j' ) ) {
                std::string temp = dp -> d_name;
                std::cout << temp << std::endl;
                imageSet.push_back( { temp, 1001 } );
                testcases.push_back( case_counter++ );
            }
        }
        closedir( directory );


        std::cout << case_counter << " pictures found, running inference = = =" << std::endl;

        //imageSet.push_back( { "shark.jpg", 1 } );

        // get all jpg in current folder
        //std::string path = "./";
        //for( const auto & entry : std::filesystem::directory_iterator( path ) ){
        //    std::cout << entry.path() << std::endl;
        //}

        // Coverity fix: The following code may throw an exception of type std::length_error.
        armnn::TensorShape inputTensorShape({ 1, 3, 224, 224 });

        using DataType = float;
        using DatabaseType = ImagePreprocessor<float>;
        using ParserType = armnnOnnxParser::IOnnxParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                     argc, argv,
                     "mobilenetv2-1.0.onnx", // model name
                     true,                           // model is binary
                     "data", "mobilenetv20_output_flatten0_reshape0", // input and output tensor names
                     testcases,                    // test images to test with as above
                     [&imageSet](const char* dataDir, const ModelType&) {
                         // This creates create a 1, 3, 224, 224 normalized input with mean and stddev to pass to Armnn
                         return DatabaseType(
                             dataDir,
                             224,
                             224,
                             imageSet,
                             255.0,                           // scale
                             {{0.485f, 0.456f, 0.406f}},      // mean
                             {{0.229f, 0.224f, 0.225f}},      // stddev
                             DatabaseType::DataFormat::NCHW); // format
                     },
                     &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: OnnxMobileNet-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
