/**
 * @file      hdf5.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @author    David Bayer \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            ibayer@fit.vutbr.cz
 *
 * @brief     PC lab 7 / HDF5
 *
 * @version   2023
 *
 * @date      05 April     2020, 12:10 (created) \n
 * @date      04 April     2023, 12:22 (revised) \n
 * @date      05 April     2024, 09:00 (revised) \n
 *
 */

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <mpi.h>
#include <hdf5.h>

#if !H5_HAVE_PARALLEL
#error "Parallel HDF5 is required"
#endif

#define DEBUG

//--------------------------------------------------------------------------------------------------------------------//
//                                                Macros and constants                                                //
//--------------------------------------------------------------------------------------------------------------------//

/// Check HDF5 call and abort if it fails.
#define CHECK_HDF5_CALL(call) checkHdf5Call(call, __FILE__, __LINE__)

/// Check HDF5 handle and abort if it is invalid.
#define CHECK_HDF5_ID(hid) checkHdf5Id(hid, __FILE__, __LINE__)

/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                            Helper function prototypes                                              //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
template <typename... Args>
void mpiPrintf(int who, const std::string_view format, Args &&...args);
/// Flush standard output.
void mpiFlush();
/// Parse command line parameters.
int parseParameters(int argc, char **argv);

/// Return MPI rank in a given communicator.
int mpiGetCommRank(MPI_Comm comm);
/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);

/// Execute a given command in the shell and return the output.
std::string exec(const std::string_view cmd);

/// Check HDF5 call and abort if it fails.
herr_t checkHdf5Call(herr_t status, const std::string_view file, int line);

/// Check HDF5 handle and abort if it is invalid.
hid_t checkHdf5Id(hid_t hid, const std::string_view file, int line);

//--------------------------------------------------------------------------------------------------------------------//
//                                                 Helper functions                                                   //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * C printf routine with selection which rank prints.
 * @tparam Args  - variable number of arguments.
 * @param who    - which rank should print. If -1 then all prints.
 * @param format - format string.
 * @param ...    - other parameters.
 */
template <typename... Args>
void mpiPrintf(int who, const std::string_view format, Args &&...args)
{
  if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD)))
  {
    if constexpr (sizeof...(args) == 0)
    {
      std::printf("%s", std::data(format));
    }
    else
    {
      std::printf(std::data(format), std::forward<Args>(args)...);
    }
  }
} // end of mpiPrintf

//----------------------------------------------------------------------------------------------------------------------

/**
 * Flush stdout and call barrier to prevent message mixture.
 */
void mpiFlush()
{
  std::fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);

  // A known hack to correctly order writes
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
} // end of mpiFlush
//----------------------------------------------------------------------------------------------------------------------

/**
 * Parse commandline - expecting a single parameter - test Id.
 * @param argc
 * @param argv
 * @return test id
 */
int parseParameters(int argc, char **argv)
{
  if (argc != 2)
  {
    mpiPrintf(MPI_ROOT_RANK, "!!!                   Please specify test number!                !!!\n");
    mpiPrintf(MPI_ROOT_RANK, "--------------------------------------------------------------------\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return std::stoi(argv[1]);
} // end of parseParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(MPI_Comm comm)
{
  int rank{};

  MPI_Comm_rank(comm, &rank);

  return rank;
} // end of mpiGetCommRank
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(MPI_Comm comm)
{
  int size{};

  MPI_Comm_size(comm, &size);

  return size;
} // end of mpiGetCommSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Execute command in shell.
 * @param [in] command line
 * @return Standard output form the commandline.
 */
std::string exec(const std::string_view cmd)
{
  struct PipeDeleter
  {
    void operator()(FILE *p) const
    {
      if (p != nullptr)
      {
#ifdef WIN32
        _pclose(p);
#else
        pclose(p);
#endif
      }
    }
  };

  auto makePipe = [](const std::string_view cmd, const std::string_view mode)
  {
#ifdef WIN32
    return _popen(cmd.data(), mode.data());
#else
    return popen(cmd.data(), mode.data());
#endif
  };

  std::array<char, 128> buffer{};
  std::string result{};

  std::unique_ptr<FILE, PipeDeleter> pipe(makePipe(cmd, "r"));

  if (pipe == nullptr)
  {
    throw std::runtime_error("popen() failed!");
  }

  // Read the commnadline output
  while (std::fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
  {
    result += buffer.data();
  }

  return result;
} // end of exec
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check HDF5 call and abort if it fails.
 * @param status - HDF5 status.
 * @param file   - file name.
 * @param line   - line number.
 */
herr_t checkHdf5Call(herr_t status, const std::string_view file, int line)
{
  if (status < 0)
  {
    mpiPrintf(MPI_ROOT_RANK, "HDF5 call failed in %s:%d\n", file, line);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return status;
} // end of checkHdf5Call
//----------------------------------------------------------------------------------------------------------------------

/**
 * Check HDF5 handle and abort if it is invalid.
 * @param hid  - HDF5 handle.
 * @param file - file name.
 * @param line - line number.
 */
hid_t checkHdf5Id(hid_t hid, const std::string_view file, int line)
{
  if (hid == H5I_INVALID_HID)
  {
    mpiPrintf(MPI_ROOT_RANK, "HDF5 invalid handle in %s:%d\n", file, line);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  return hid;
} // end of checkHdf5Id
//----------------------------------------------------------------------------------------------------------------------

/**
 * Initialize matrix.
 * @tparam T               - Datatype of the matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
template <typename T>
void initMatrix(T *matrix,
                const int nRows,
                const int nCols)
{
  for (int i = 0; i < nRows; i++)
  {
    for (int j = 0; j < nCols; j++)
    {
      matrix[i * nCols + j] = static_cast<T>(100 * i + j);
    }
  }
} // end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix
 * @tparam T         - Datatype of the matrix.
 * @param  [in, out] matrix - Block to print out.
 * @param  [in]      nRows  - Number of rows in the block.
 * @param  [in]      nCols  - Number of cols.
 */
template <typename T>
void printMatrix(T *matrix,
                 int nRows,
                 int nCols)
{
  std::string str{};

  for (int i = 0; i < nRows; i++)
  {
    std::array<char, 16> buffer{};

    str.clear();

    for (int j = 0; j < nCols; j++)
    {
      if constexpr (std::is_floating_point_v<T>)
      {
        std::snprintf(buffer.data(), buffer.size(), "%s%6.3f", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
      }
      else
      {
        std::snprintf(buffer.data(), buffer.size(), "%s%8d", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
      }

      str += buffer.data();
    }

    std::printf(" - Rank %2d = [%s]\n", mpiGetCommRank(MPI_COMM_WORLD), str.c_str());
  }
} // end of printMatrix
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//                                                   Main routine                                                     //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Main function.
 */
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // Set stdout to print out
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 7                                \n");
  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiFlush();

  // Parse parameters
  const int testId = parseParameters(argc, argv);
  // Select test
  switch (testId)
  {
    //--------------------------------------------------------------------------------------------------------------------//
    //                        Example 1 - Create a HDF5 file and write a scalar from the root rank                        //
    //--------------------------------------------------------------------------------------------------------------------//
  case 1:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 1 - Create a HDF5 file and write a scalar from the root rank\n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // The first example only uses the root rank (serial IO). We first create a HDF5 file, then create a dataset and
    // finally write a scalar value of 128 into it.

    constexpr std::string_view fileName{"File1.h5"};
    constexpr std::string_view datasetName{"Dataset-1"};

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      // 1. Declare an HDF5 file.
      H5open();

      hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

      mpiPrintf(MPI_ROOT_RANK, " Creating file...\n");
      // 2. Create a file with write permission. Use such a flag that overrides existing file.
      //    The list of flags is in the header file called H5Fpublic.h

      hid_t file_id = H5Fcreate(fileName.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

      H5Pclose(plist_id);

      // 3. Create file and memory spaces. We will only write a single value.

      hsize_t dims[1] = {1};

      hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

      hid_t dataset_id = H5Dcreate2(file_id, datasetName.data(), H5T_NATIVE_INT, H5Screate(H5S_SCALAR), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      // 4. Create a dataset of a size [1] and int datatype.
      //    The list of predefined datatypes can be found in H5Tpublic.h

      constexpr int value{128};

      mpiPrintf(MPI_ROOT_RANK, " Writing scalar value...\n");
      // 5. Write value into the dataset.

      H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);

      // 6. Close dataset.

      H5Dclose(dataset_id);

      // 7. Close memspace.

      H5Sclose(dataspace_id);

      // 8. Close filespace.

      H5Fclose(file_id);

      mpiPrintf(MPI_ROOT_RANK, " Closing file...\n");
      // 9. Close file
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 8. Use command line tools h5ls and h5dump to see what's in the file.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, "h5ls output:\n");

      std::array<char, 256> cmd{};

      std::snprintf(cmd.data(), cmd.size(), "h5ls -f -r  %s", fileName.data());
      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");

      mpiPrintf(MPI_ROOT_RANK, "h5dump output:\n");
      std::snprintf(cmd.data(), cmd.size(), "h5dump -p -d %s  %s", datasetName.data(), fileName.data());

      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
    }

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 1

    //--------------------------------------------------------------------------------------------------------------------//
    //                                  Example 2 - Write a matrix distributed over rows                                  //
    //--------------------------------------------------------------------------------------------------------------------//
  case 2:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 2 - Write a matrix distributed over rows                    \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // File name and dataset name
    constexpr std::string_view filename{"File2.h5"};
    constexpr std::string_view datasetname{"matrix"};

    // 16 x 4 matrix
    constexpr int nRows = 16;
    constexpr int nCols = 4;

    // Distribution
    const int lRows = nRows / mpiGetCommSize(MPI_COMM_WORLD);

    // global matrix in the root
    std::vector<int> gMatrix{};
    // local stripe on each rank
    std::vector<int> lMatrix{};

    // Initialize matrix on the root
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      gMatrix.resize(nRows * nCols);

      initMatrix(gMatrix.data(), nRows, nCols);

      std::puts("Original array:");
      printMatrix(gMatrix.data(), nRows, nCols);
      std::putchar('\n');
    }

    lMatrix.resize(lRows * nCols);

    MPI_Datatype rowType{MPI_DATATYPE_NULL};
    MPI_Type_contiguous(nCols, MPI_INT, &rowType);
    MPI_Type_commit(&rowType);

    // Scatter matrix over rows
    MPI_Scatter(gMatrix.data(), lRows, rowType, lMatrix.data(), lRows, rowType, 0, MPI_COMM_WORLD);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Let's have a matrix nRows * nCols scattered by rows onto ranks. Each rank maintains lRows * nCols
    // The goal is to create a dataset in the HDF5 file and write the matrix using collective IO there.

    // 1. Declare an HDF5 file.
    H5open();

    // 2. Create a property list to open the file using MPI-IO in the MPI_COMM_WORLD communicator.

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);

    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    mpiPrintf(MPI_ROOT_RANK, " Creating file...\n");
    // 3. Create a file called (filename) with write permission. Use such a flag that overrides existing file.
    //    The list of flags is in the header file called H5Fpublic.h

    hid_t file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

    // 4. Close file access list.

    H5Pclose(plist_id);

    // 5. Create file space - a 2D matrix [nRows][nCols]
    //    Create mem space  - a 2D matrix [lRows][nCols] mapped on 1D array lMatrix.

    hsize_t dims[2] = {nRows, nCols};
    hsize_t ldims[2] = {(hsize_t)lRows, nCols};

    hid_t filespace_id = H5Screate_simple(2, dims, NULL);
    hid_t memspace_id = H5Screate_simple(2, ldims, NULL);

    mpiPrintf(MPI_ROOT_RANK, " Creating dataset...\n");
    // 6. Create a dataset. The name is store in datasetname, datatype is int. All other parameters are default.

    hid_t dataset_id = H5Dcreate2(file_id, datasetname.data(), H5T_NATIVE_INT, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    mpiPrintf(MPI_ROOT_RANK, " Selecting hyperslab...\n");
    // 7. Select a hyperslab to write a local submatrix into the dataset.

    hsize_t start[2] = {(hsize_t)(lRows * mpiGetCommRank(MPI_COMM_WORLD)), 0};
    hsize_t count[2] = {(hsize_t)lRows, nCols};

    H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);

    // 8. Create XFER property list and set Collective IO.

    hid_t xfer_plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    mpiPrintf(MPI_ROOT_RANK, " Writing data...\n");
    // 9. Write data into the dataset.

    H5Dwrite(dataset_id, H5T_NATIVE_INT, memspace_id, filespace_id, xfer_plist_id, lMatrix.data());

    // 10. Close XREF property list.

    H5Pclose(xfer_plist_id);

    // 11. Close dataset.

    H5Dclose(dataset_id);

    // 12. Close memspace and filespace.

    H5Sclose(memspace_id);
    H5Sclose(filespace_id);

    mpiPrintf(MPI_ROOT_RANK, " Closing file...\n");
    // 13. Close dataset.

    H5Fclose(file_id);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    MPI_Type_free(&rowType);

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, "h5ls output:\n");

      std::array<char, 256> cmd{};

      std::snprintf(cmd.data(), cmd.size(), "h5ls -f -r  %s", filename.data());
      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");

      mpiPrintf(MPI_ROOT_RANK, "h5dump output:\n");
      std::snprintf(cmd.data(), cmd.size(), "h5dump -p -d %s  %s", datasetname.data(), filename.data());
      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
    }

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 2

    //--------------------------------------------------------------------------------------------------------------------//
    //                                         Example 3 - Hadamard product A°B'                                          //
    //--------------------------------------------------------------------------------------------------------------------//
  case 3:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 3 -  Hadamard product C = A°B' with data in a HDF5 file\n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Input file name
    constexpr std::string_view filename{"Matrix-File.h5"};

    // HDF5 groups for input and output data.
    constexpr std::string_view inputGroupName{"Inputs"};
    constexpr std::string_view outputGroupName{"Outputs"};

    // Matrix A name
    constexpr std::string_view matrixAName{"Matrix-A"};
    // Matrix B name
    constexpr std::string_view matrixBName{"Matrix-B"};
    // Matrix C name
    constexpr std::string_view matrixCrefName{"Matrix-C-ref"};

    /**
     * @class Dim2
     * @brief 2D dimension sizes
     */
    class Dim2
    {
    public:
      /// Default constructor.
      constexpr Dim2() noexcept = default;
      /// Initializing constructor.
      constexpr Dim2(hsize_t nRows, hsize_t nCols) noexcept : mData{nRows, nCols} {}
      /// Array -> Dims
      constexpr Dim2(const hsize_t dims[]) : Dim2{dims[0], dims[1]} {}

      /// Get number of elements
      constexpr std::size_t nElements() const noexcept { return static_cast<std::size_t>(mData[0] * mData[1]); }
      /// Convert dimensions to string
      std::string toString() const
      {
        std::array<char, 100> str{};

        std::snprintf(str.data(), str.size(), "[%lld, %lld]", mData[0], mData[1]);

        return std::string{str.data()};
      }

      /// Get dimensions as an array
      constexpr const hsize_t *data() const noexcept { return mData.data(); }

      /// Get dimensions as an array
      constexpr hsize_t *data() noexcept { return mData.data(); }

      /// Get number of dimensions
      constexpr std::size_t size() const noexcept { return mData.size(); }

      /// Get number of rows
      constexpr hsize_t nRows() const noexcept { return mData[0]; }
      constexpr hsize_t y() const noexcept { return mData[0]; }

      /// Get number of columns
      constexpr hsize_t nCols() const noexcept { return mData[1]; }
      constexpr hsize_t x() const noexcept { return mData[1]; }

    private:
      /// Dimensions saved in row major order.
      std::array<hsize_t, 2> mData{};
    };

    // Print content of the matrix
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      mpiPrintf(MPI_ROOT_RANK, "h5ls output:\n");

      std::array<char, 256> cmd{};

      // List the content of the file
      std::snprintf(cmd.data(), cmd.size(), "h5ls -f -r  %s", filename.data());
      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");

      // If DEBUG then print both header and data.
#ifdef DEBUG
      mpiPrintf(MPI_ROOT_RANK, "h5dump output:\n");
      sprintf(cmd.data(), "h5dump %s ", filename.data());
#else
      mpiPrintf(MPI_ROOT_RANK, "h5dump output (dataset data omitted, enable by using -d parameter):\n");
      std::snprintf(cmd.data(), cmd.size(), "h5dump -H %s ", filename.data());
#endif
      mpiPrintf(MPI_ROOT_RANK, "%s\n", exec(cmd.data()).c_str());
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // The goal of this exercise is to read two matrices from the input file and calculate the Hadamard product
    // of A and B transposed C = A ° B'. Hadamard product is a simple element wise multiplication.
    //  C = A ° B' ; for each i, j: c[i][j] = a[i][j] * b[j][i].
    //
    // To make it fast, we will distribute matrices A and C by row blocks, while the matrix B by column blocks.
    // The file also contains a reference output we can compare to.

    // Local part of matrices you need.
    std::vector<double> matrixA{};
    std::vector<double> matrixB{};
    // Matrix to store my result.
    std::vector<double> matrixC{};
    // Reference matrix in the file.
    std::vector<double> matrixCref{};

    // 1. Create a property list to open the HDF5 file using MPI-IO in the MPI_COMM_WORLD communicator.

    H5open();

    hid_t plist_id = CHECK_HDF5_ID(H5Pcreate(H5P_FILE_ACCESS));

    H5Pset_fapl_mpio(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL);

    mpiPrintf(MPI_ROOT_RANK, " Opening file...\n");
    // 2. Open a file called (filename) with read-only permission.
    //    The list of flags is in the header file called H5Fpublic.h

    hid_t file_id = CHECK_HDF5_ID(H5Fopen(filename.data(), H5F_ACC_RDONLY, plist_id));

    // 3. Close file access list.

    H5Pclose(plist_id);

    // 4. Open HDF5 groups with input and output matrices.

    hid_t group_input = CHECK_HDF5_ID(H5Gopen2(file_id, inputGroupName.data(), H5P_DEFAULT));
    hid_t group_output = CHECK_HDF5_ID(H5Gopen2(file_id, outputGroupName.data(), H5P_DEFAULT));

    // 5. Open HDF5 datasets with input and output matrices.

    hid_t datasetA = CHECK_HDF5_ID(H5Dopen2(group_input, matrixAName.data(), H5P_DEFAULT));
    hid_t datasetB = CHECK_HDF5_ID(H5Dopen2(group_input, matrixBName.data(), H5P_DEFAULT));
    hid_t datasetCref = CHECK_HDF5_ID(H5Dopen2(group_output, matrixCrefName.data(), H5P_DEFAULT));

    // 6. Write a lambda function to read the dataset size. The routine takes dataset ID and returns Dims2D.
    //   i.   Get the dataspace from the dataset using H5Dget_space.
    //   ii.  Read dataset sizes using H5Sget_simple_extent_dims.
    //        The rank of the dataspace is 2 (2D matrices) and the dimensions are stored in Row, Col manner (row major)
    //        You can use a conversion dims.toArray() to pass the structure as an array.
    //   iii. Close the dataspace.
    auto getDims = [](hid_t dataset) -> Dim2
    {
      Dim2 dims{};
      hid_t dataspace_id = H5Dget_space(dataset);
      hsize_t dims_out[2];
      H5Sget_simple_extent_dims(dataspace_id, dims_out, NULL);
      H5Sclose(dataspace_id);
      return Dim2(dims_out);
    }; // end of getDims

    mpiPrintf(MPI_ROOT_RANK, " Reading dimension sizes...\n");
    // 7. Get global matrix dimension sizes.
    const Dim2 gDimsA = getDims(datasetA);
    const Dim2 gDimsB = getDims(datasetB);
    const Dim2 gDimsC = getDims(datasetCref);

    // 8. Calculate local matrix dimension sizes. A, C and Cref are distributed by rows, B by columns.
    const Dim2 lDimsA = {gDimsA.nRows() / mpiGetCommSize(MPI_COMM_WORLD), gDimsA.nCols()};
    const Dim2 lDimsB = {gDimsB.nRows(), gDimsB.nCols() / mpiGetCommSize(MPI_COMM_WORLD)};
    const Dim2 lDimsC = {gDimsC.nRows() / mpiGetCommSize(MPI_COMM_WORLD), gDimsC.nCols()};

    // Print out dimension sizes
    mpiPrintf(MPI_ROOT_RANK, "  - Number of ranks: %d\n", mpiGetCommSize(MPI_COMM_WORLD));
    mpiPrintf(MPI_ROOT_RANK, "  - Matrix A global and local size [%llu, %llu] / [%llu, %llu]\n",
              gDimsA.nRows(), gDimsA.nCols(), lDimsA.nRows(), lDimsA.nCols());
    mpiPrintf(MPI_ROOT_RANK, "  - Matrix B global and local size [%llu, %llu] / [%llu, %llu]\n",
              gDimsB.nRows(), gDimsB.nCols(), lDimsB.nRows(), lDimsB.nCols());
    mpiPrintf(MPI_ROOT_RANK, "  - Matrix C global and local size [%llu, %llu] / [%llu, %llu]\n",
              gDimsC.nRows(), gDimsC.nCols(), lDimsC.nRows(), lDimsC.nCols());

    // Allocate memory for local arrays
    matrixA.resize(lDimsA.nElements());
    matrixB.resize(lDimsB.nElements());
    matrixC.resize(lDimsC.nElements());
    matrixCref.resize(lDimsC.nElements());

    // 9. Write a lambda function to read a particular slab at particular ranks.
    //    i.   Create a 2D filespace, then select the part of the dataset you want to read.
    //    ii.  Create a memspace where to read the data.
    //    iii. Enable collective MPI-IO.
    //    iv.  Read the slab.
    //    v.   Close property list, memspace and filespace.
    auto readSlab = [](hid_t dataset,
                       const Dim2 &slabStart, const Dim2 &slabSize, const Dim2 &datasetSize,
                       double *data) -> void
    {
      // Create file dataspace and select a hyperslab
      hid_t file_space = H5Dget_space(dataset);
      hsize_t start[2] = {slabStart.y(), slabStart.x()}; // Start of slab
      hsize_t count[2] = {1, 1};                         // Block count
      hsize_t stride[2] = {1, 1};                        // Block stride
      hsize_t block[2] = {slabSize.y(), slabSize.x()};   // Block size
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, stride, count, block);

      // Create memory dataspace
      hsize_t mem_dims[2] = {slabSize.y(), slabSize.x()};
      hid_t mem_space = H5Screate_simple(2, mem_dims, NULL);

      // Enable collective MPI-IO
      hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

      // Read the slab
      H5Dread(dataset, H5T_NATIVE_DOUBLE, mem_space, file_space, plist_id, data);

      // Cleanup
      H5Pclose(plist_id);
      H5Sclose(mem_space);
      H5Sclose(file_space);
    }; // end of readSlab

    // 10. Read parts of the matrices A, B and Cref.

    mpiPrintf(MPI_ROOT_RANK, "  Reading matrix A...\n");
    readSlab(datasetA, {lDimsA.nRows() * mpiGetCommRank(MPI_COMM_WORLD), 0}, lDimsA, gDimsA, matrixA.data());

    mpiPrintf(MPI_ROOT_RANK, "  Reading matrix B...\n");
    readSlab(datasetB, {0, lDimsB.nCols() * mpiGetCommRank(MPI_COMM_WORLD)}, lDimsB, gDimsB, matrixB.data());

    mpiPrintf(MPI_ROOT_RANK, "  Reading matrix Cref...\n");
    readSlab(datasetCref, {lDimsC.nRows() * mpiGetCommRank(MPI_COMM_WORLD), 0}, lDimsC, gDimsC, matrixCref.data());

    mpiPrintf(MPI_ROOT_RANK, "  Calculating C = A ° B' ...\n");

    // Calculate the Hadamard product C = A ° B'
    std::fill(matrixC.begin(), matrixC.end(), 0.0);

    for (hsize_t row{}; row < lDimsC.nRows(); row++)
    {
      for (hsize_t col{}; col < lDimsC.nCols(); col++)
      {
        matrixC[row * lDimsC.nCols() + col] = matrixA[row * lDimsA.nCols() + col] * matrixB[col * lDimsB.nCols() + row];
      }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef DEBUG
    mpiPrintf(MPI_ROOT_RANK, " matrixA [%lld, %lld]\n", lDimsA.nRows(), lDimsA.nCols());
    printMatrix(matrixA.data(), lDimsA.nRows(), lDimsA.nCols());
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    mpiPrintf(MPI_ROOT_RANK, " matrixB [%lld, %lld]\n", lDimsB.nRows(), lDimsB.nCols());
    printMatrix(matrixB.data(), lDimsB.nRows(), lDimsB.nCols());
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    mpiPrintf(MPI_ROOT_RANK, " matrixC [%lld, %lld]\n", lDimsC.nRows(), lDimsC.nCols());
    printMatrix(matrixC.data(), lDimsC.nRows(), lDimsC.nCols());
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
#endif
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiPrintf(MPI_ROOT_RANK, "  Verification C == Cref ...\n");

    // Validate data C == Cref and print out maximum absolute error.
    double globalMaxError{};
    double localMaxError{};

    for (std::size_t i{}; i < lDimsC.nElements(); i++)
    {
      localMaxError = std::max(localMaxError, std::abs(matrixCref[i] - matrixC[i]));
    }

    MPI_Reduce(&localMaxError, &globalMaxError, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    mpiPrintf(MPI_ROOT_RANK, "  Maximum abs error = %e\n", globalMaxError);

    // 11. Close datasets
    H5Dclose(datasetA);
    H5Dclose(datasetB);
    H5Dclose(datasetCref);

    // 12. Close HDF5 groups
    H5Gclose(group_input);
    H5Gclose(group_output);

    // 13. Close HDF5 file
    H5Fclose(file_id);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    break;
  } // case 3

  default:
  {
    mpiPrintf(0, " !!!                     Unknown test number                     !!!\n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    break;
  }
  } // switch

  MPI_Finalize();
} // end of main
//----------------------------------------------------------------------------------------------------------------------
