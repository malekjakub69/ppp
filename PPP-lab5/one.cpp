/**
 * @file      one.cpp
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
 * @brief     PC lab 5 / MPI one-sided communications
 *
 * @version   2023
 *
 * @date      21 March     2020, 13:05 (created) \n
 * @date      21 March     2023, 11:48 (created) \n
 * @date      20 March     2023, 07:00 (revised) \n
 *
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

#include <mpi.h>

/// constant used for mpi printf
constexpr int MPI_ALL_RANKS = -1;
/// constant for root rank
constexpr int MPI_ROOT_RANK = 0;

//--------------------------------------------------------------------------------------------------------------------//
//                                            Helper function prototypes                                              //
//--------------------------------------------------------------------------------------------------------------------//
/// Wrapper to the C printf routine and specifies who can print (a given rank or all).
template <typename... Args>
void mpiPrintf(int who, const std::string_view format, Args... args);
/// Flush standard output.
void mpiFlush();
/// Parse command line parameters.
int parseParameters(int argc, char **argv);

/// Return MPI rank in a given communicator.
int mpiGetCommRank(MPI_Comm comm);
/// Return size of the MPI communicator.
int mpiGetCommSize(MPI_Comm comm);

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
void mpiPrintf(int who, const std::string_view format, Args... args)
{
  if ((who == MPI_ALL_RANKS) || (who == mpiGetCommRank(MPI_COMM_WORLD)))
  {
    if constexpr (sizeof...(args) == 0)
    {
      std::printf("%s", std::data(format));
    }
    else
    {
      std::printf(std::data(format), args...);
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
 * Initialize matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
void initMatrix(int *matrix,
                int nRows,
                int nCols)
{
  for (int i = 0; i < nRows; i++)
  {
    for (int j = 0; j < nCols; j++)
    {
      matrix[i * nCols + j] = mpiGetCommRank(MPI_COMM_WORLD) * 10000 + i * 100 + j;
    }
  }
} // end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Clear matrix.
 * @param [in, out] matrix - Matrix to initialize.
 * @param [in]      nRows  - Number of rows.
 * @param [in]      nCols  - Number of cols.
 */
void clearMatrix(int *matrix,
                 int nRows,
                 int nCols)
{
  std::fill_n(matrix, nRows * nCols, 0);
} // end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix.
 * @param block     - Block to print out
 * @param blockSize - Number of rows in the block
 * @param nCols     - Number of cols.
 */
void printMatrix(int *matrix,
                 int nRows,
                 int nCols)
{
  static constexpr std::size_t maxTmpLen{64};

  const int rank = mpiGetCommRank(MPI_COMM_WORLD);

  std::string str{};
  char val[maxTmpLen]{};

  for (int i = 0; i < nRows; i++)
  {
    str.clear();

    for (int j = 0; j < nCols; j++)
    {
      std::sprintf(val, "%s%8d", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
      str += val;
    }

    std::printf(" - Rank %2d, row %2d = [%s]\n", rank, i, str.c_str());
  }
} // end of printBlock
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print block of the matrix.
 * @param block     - Block to print out
 * @param blockSize - Number of rows in the block
 * @param nCols     - Number of cols.
 */
void printMatrix(double *matrix,
                 int nRows,
                 int nCols)
{
  static constexpr std::size_t maxTmpLen{64};

  const int rank = mpiGetCommRank(MPI_COMM_WORLD);

  std::string str{};
  char val[maxTmpLen]{};

  for (int i = 0; i < nRows; i++)
  {
    str.clear();

    for (int j = 0; j < nCols; j++)
    {
      std::sprintf(val, "%s%6.3f", (j == 0) ? "" : ", ", matrix[i * nCols + j]);
      str += val;
    }

    std::printf(" - Rank %2d, row %2d = [%s]\n", rank, i, str.c_str());
  }
} // end of printBlock
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//                                                   Main routine                                                     //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Main function
 */
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  // Set stdout to print out
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 5                                \n");
  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiFlush();

  // Parse parameters
  const int testId = parseParameters(argc, argv);
  // const int testId = 4;

  // Select test
  switch (testId)
  {
    //--------------------------------------------------------------------------------------------------------------------//
    //                            Example 1 - Implement matrix transposition using Put and Get                            //
    //--------------------------------------------------------------------------------------------------------------------//
  case 1:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 1 - Implement matrix transposition using Put and Get        \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Having P ranks, we have a distributed matrix P x P.
    const int nRows = mpiGetCommSize(MPI_COMM_WORLD);
    const int nCols = mpiGetCommSize(MPI_COMM_WORLD);

    // Every rank has one row of this distributed matrix.
    int *distMatrix{};
    // term matrix is local scratch place.
    std::vector<int> tempRow{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // The goal is to transpose a distributed matrix using Put and Get.
    // Since we cannot guarantee the ordering, we will have to run the algorithm in multiple phases.
    // First, we collect appropriate columns in tempRow at particular rank.
    // Second, we copy the row back to distMatrix.
    // Third, we print out transposed distMatrix.
    // Finally, we use an appropriate collective communication to transpose distMatrix back.

    //*** In order to synchronize, you'll have to use several fences at appropriate places. ***

    // 1. Declare an MPI window for the distMatrix.

    MPI_Win win;

    // 2. Allocate tempRow and distMatrix. For the distMatrix, use an appropriate MPI function.

    tempRow.resize(nCols);
    MPI_Win_allocate(sizeof(int) * nCols, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &distMatrix, &win);

    // Clear tempMatrix and init and print distMatrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "------------------------ Original matrix ----------------------------\n");

    mpiFlush();
    clearMatrix(tempRow.data(), 1, nCols);
    initMatrix(distMatrix, 1, nCols);
    printMatrix(distMatrix, 1, nCols);

    // 3. Create a window. Every rank exposed its row.

    MPI_Win_create(distMatrix, sizeof(int) * nCols, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // 4. Every rank collects appropriate column from all ranks into tempRow.
    //    Rank 0 collects the 1st col, rank 1 the 2nd col, etc.
    //    Use MPI_Get operations.

    MPI_Win_fence(0, win);

    for (int i = 0; i < nCols; i++)
    {
      MPI_Get(&tempRow[i], 1, MPI_INT, i, mpiGetCommRank(MPI_COMM_WORLD), 1, MPI_INT, win);
    }

    MPI_Win_fence(0, win);

    // 5. Copy the row back into the window (although we could use local copy, we'll use MPI_Put).

    MPI_Put(tempRow.data(), nCols, MPI_INT, mpiGetCommRank(MPI_COMM_WORLD), 0, nCols, MPI_INT, win);

    MPI_Win_fence(0, win);

    // Print transposed matrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "----------------------- Transposed matrix ---------------------------\n");
    mpiFlush();
    printMatrix(distMatrix, 1, nCols);

    // 5. Use a collective communication to transpose the matrix back.

    MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT, distMatrix, 1, MPI_INT, MPI_COMM_WORLD);

    // Print original matrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "------------------------ Original matrix ----------------------------\n");
    mpiFlush();
    printMatrix(distMatrix, 1, nCols);

    // 6. Free memory, the window, etc.

    MPI_Win_free(&win);
    MPI_Free_mem(distMatrix);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 1

    //--------------------------------------------------------------------------------------------------------------------//
    //                       Example 2 - Matrix transposition using Put and Get and MPI_Info params                       //
    //--------------------------------------------------------------------------------------------------------------------//
  case 2:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 2 - Implement matrix transposition using Put and Get        \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Having P ranks, we have a distributed matrix P x P.
    const int nRows = mpiGetCommSize(MPI_COMM_WORLD);
    const int nCols = mpiGetCommSize(MPI_COMM_WORLD);

    // Every rank has one row of this distributed matrix.
    int *distMatrix{};
    // term matrix is the local scratch place.
    std::vector<int> tempRow{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Use the implementation from the previous step and add:
    // - parameters to the window (same_size, same_disp_unit)
    // - asserts to the fences, e.g., the there was no communication before the first one, no after the last one, etc.

    //*** In order to synchronize, you'll have to use several fences at appropriate places. ***

    // 1. Declare an MPI window for the distMatrix.

    MPI_Win win;

    // 2. Allocate tempRow and distMatrix. For the distMatrix, use an appropriate MPI function.

    tempRow.resize(nCols);
    MPI_Win_allocate(sizeof(int) * nCols, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &distMatrix, &win);

    // Clear tempMatrix and init and print distMatrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "------------------------ Original matrix ----------------------------\n");

    mpiFlush();
    clearMatrix(tempRow.data(), 1, nCols);
    initMatrix(distMatrix, 1, nCols);
    printMatrix(distMatrix, 1, nCols);

    // 3. Declare and create MPI_Info object.

    MPI_Info winInfo;

    MPI_Info_create(&winInfo);

    // 4. Set info parameters to the window (same_size, same_disp_unit).

    MPI_Info_set(winInfo, "same_size", "true");
    MPI_Info_set(winInfo, "same_disp_unit", "true");

    // 5. Create a window. Every rank exposes its row.

    MPI_Win_create(distMatrix, sizeof(int) * nCols, sizeof(int), winInfo, MPI_COMM_WORLD, &win);

    // 6. Free winInfo object.

    MPI_Info_free(&winInfo);

    //--------------------------------------------------------------------------------------------------------------//

    // 7. Every rank collects appropriate column from all ranks into tempRow.
    //    Rank 0 collects the 1st col, rank 1 the 2nd col, etc.
    //    Use MPI_Get operations {MPI_MODE_NOSTORE, MPI_MODE_NOPUT, MPI_MODE_NOPRECEDE, MPI_MODE_NOSUCCEED}.
    //    You can use multiple by operator |, eg. MPI_MODE_NOSTORE | MPI_MODE_NOPUT

    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT | MPI_MODE_NOSTORE, win);

    for (int i = 0; i < nCols; i++)
    {
      MPI_Get(&tempRow[i], 1, MPI_INT, i, mpiGetCommRank(MPI_COMM_WORLD), 1, MPI_INT, win);
    }

    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT, win);

    // 8. Copy the row back into the window (although we could use local copy, we'll use MPI_Put).

    MPI_Put(tempRow.data(), nCols, MPI_INT, mpiGetCommRank(MPI_COMM_WORLD), 0, nCols, MPI_INT, win);

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    // Print transposed matrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "----------------------- Transposed matrix ---------------------------\n");
    mpiFlush();
    printMatrix(distMatrix, 1, nCols);

    // 9. Use a collective communication to transpose the matrix back.

    MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT, distMatrix, 1, MPI_INT, MPI_COMM_WORLD);

    // Print original matrix.
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "------------------------ Original matrix ----------------------------\n");
    mpiFlush();
    printMatrix(distMatrix, 1, nCols);

    // 10. Free memory, the window, etc.

    MPI_Win_free(&win);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 2

    //--------------------------------------------------------------------------------------------------------------------//
    //                                         Example 3 - Distributed histogram                                          //
    //--------------------------------------------------------------------------------------------------------------------//
  case 3:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 3 - Implement distributed histogram using MPI_Accumulate.   \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // We have an array of 32M elements (128 MB).
    constexpr int gArraySize = 1024 * 1024 * 32;
    const int lArraySize = gArraySize / mpiGetCommSize(MPI_COMM_WORLD);

    // The histogram has 128 bins.
    constexpr int gHistSize = 128;
    const int lHistSize = gHistSize / mpiGetCommSize(MPI_COMM_WORLD);

    // Simple hash function to make the histogram more computationally intensive.
    auto hash = [gHistSize](int value)
    {
      int tmp = value;

      for (int i = 0; i < 120; i++)
      {
        tmp = ((tmp * 3671) % 127 + tmp);
      }
      return tmp % gHistSize;
    };

    // Print histogram
    auto printHist = [](int *hist, int size)
    {
      constexpr int elemsPerLine = 16;

      for (int i = 0; i < size; i += elemsPerLine)
      {
        std::printf("bin = %3d: [", i);

        for (int j = 0; j < elemsPerLine; j++)
        {
          std::printf("%s%6d", (j == 0) ? "" : ", ", hist[i + j]);
        }

        std::printf("]\n");
      }
    };

    // gArray is the global array in the root.
    std::vector<int> gArray{};
    // lArray is the local array in every rank Scatter( gArray -> lArray);
    std::vector<int> lArray{};

    // Distributed histogram, every rank has a part of bins.
    int *distHist{};

    // Time measure
    double seqTime{}, parTime{}, startTime{};

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Generating data ...\n");
    mpiFlush();

    // Allocate an init input data.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      gArray.resize(gArraySize);

      std::random_device dev{};
      std::mt19937 gen{dev()};
      std::uniform_int_distribution<int> dis(0, gHistSize - 1);

      std::generate(gArray.begin(), gArray.end(), [&dis, &gen]
                    { return dis(gen); });
    }

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//
    //                                             Sequential version                                               //
    //--------------------------------------------------------------------------------------------------------------//
    startTime = MPI_Wtime();
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      std::vector<int> seqHist(gHistSize);

      // Compute histogram
      std::for_each(gArray.begin(), gArray.end(), [&seqHist, hash](int value)
                    { seqHist[hash(value)]++; });

      // Print histogram
      std::printf("Seq histogram: \n");
      printHist(seqHist.data(), gHistSize);
    }
    seqTime = MPI_Wtime() - startTime;

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // We have an array of numbers. The goal is to calculate distributed histogram.
    // The array is first scattered over all ranks. Every rank holds a part of the histogram and the others increment
    // the bins
    //   - It is not an efficient solution, of course, it's just an exercise.

    /// *** You have to place Fences at several places to prevent data races! ***

    startTime = MPI_Wtime();

    // 1. Distributed the global array (gArray) into lArray. Every rank has lArraySize elements.

    lArray.resize(lArraySize);

    MPI_Scatter(gArray.data(), lArraySize, MPI_INT, lArray.data(), lArraySize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // 2. Declare MPI window for the histogram an MPI_Info object.

    MPI_Win win;
    MPI_Info info;
    MPI_Info_create(&info);

    // 3.  Create an info set same_size and same_disp_unit to true.

    MPI_Info_set(info, "same_size", "true");
    MPI_Info_set(info, "same_disp_unit", "true");

    // 4. Allocate the window. Use the MPI_Win_allocate routine to create the window and allocate memory for it).

    MPI_Win_allocate(sizeof(int) * lHistSize, sizeof(int), info, MPI_COMM_WORLD, &distHist, &win);

    // Clear my part of the histogram, this can be done in local memory without MPI routines.
    std::fill_n(distHist, lHistSize, 0);

    // This fence is necessary to ensure the array has been created at the target an initialized.

    // 5. Use MPI_Accumulate routine to increment the appropriate bin. Hist[hash(lArray[i])]++

    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT | MPI_MODE_NOSTORE, win);

    int valueOne[] = {1};

    for (int i = 0; i < lArraySize; i++)
    {
      int hashValue = hash(lArray[i]);

      int rank = hashValue / lHistSize;
      int disp = hashValue % lHistSize;

      MPI_Accumulate(valueOne, 1, MPI_INT, rank, disp, 1, MPI_INT, MPI_SUM, win);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);

    // Print histogram
    std::vector<int> gHist{};

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      gHist.resize(gHistSize);
    }

    // 6. Collect all data from all ranks into a single histogram on the root.

    MPI_Gather(distHist, lHistSize, MPI_INT, gHist.data(), lHistSize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      std::printf("Dist histogram: \n");
      printHist(gHist.data(), gHistSize);
    }

    // 7. Free all used variables.

    MPI_Win_free(&win);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    parTime = MPI_Wtime() - startTime;

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiPrintf(MPI_ROOT_RANK, "| Seq time: %6.2fs                                                 |\n",
              seqTime);
    mpiPrintf(MPI_ROOT_RANK, "| Par time: %6.2fs (P = %2d)                                        |\n",
              parTime, mpiGetCommSize(MPI_COMM_WORLD));
    mpiPrintf(MPI_ROOT_RANK, "| Speedup:  %6.2fx (%4.2f%%)                                        |\n",
              seqTime / parTime, (seqTime / parTime) / mpiGetCommSize(MPI_COMM_WORLD) * 100);
    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiFlush();
    break;
  } // case 3

    //--------------------------------------------------------------------------------------------------------------------//
    //                       Example 4 - Distributed histogram with local storage and MPI_Accumulate                      //
    //--------------------------------------------------------------------------------------------------------------------//
  case 4:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 4 - Distributed histogram using MPI_Accumulate.             \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // We have an array of 32M elements (128 MB).
    constexpr int gArraySize = 1024 * 1024 * 32;
    const int lArraySize = gArraySize / mpiGetCommSize(MPI_COMM_WORLD);

    // The histogram has 128 bins.
    constexpr int gHistSize = 128;
    const int lHistSize = gHistSize / mpiGetCommSize(MPI_COMM_WORLD);

    // Simple hash function to make the histogram more computationally intensive.
    auto hash = [gHistSize](int value)
    {
      int tmp = value;

      for (int i = 0; i < 120; i++)
      {
        tmp = ((tmp * 3671) % 127 + tmp);
      }
      return tmp % gHistSize;
    };

    // Print histogram
    auto printHist = [](int *hist, int size)
    {
      constexpr int elemsPerLine = 16;

      for (int i = 0; i < size; i += elemsPerLine)
      {
        std::printf("bin = %3d: [", i);

        for (int j = 0; j < elemsPerLine; j++)
        {
          std::printf("%s%6d", (j == 0) ? "" : ", ", hist[i + j]);
        }

        std::printf("]\n");
      }
    };

    // gArray is the global array in the root.
    std::vector<int> gArray{};
    // lArray is the local array in every rank Scatter( gArray -> lArray);
    std::vector<int> lArray{};

    // Distributed histogram, every rank has a part of bins
    int *distHist{};
    // Local scratch place
    std::vector<int> localHist{};

    // Time measure
    double seqTime{}, parTime{}, startTime{};

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Generating data ...\n");
    mpiFlush();

    // Allocate an init input data.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      gArray.resize(gArraySize);

      std::random_device dev{};
      std::mt19937 gen{dev()};
      std::uniform_int_distribution<int> dis(0, gHistSize - 1);

      std::generate(gArray.begin(), gArray.end(), [&dis, &gen]
                    { return dis(gen); });
    }

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//
    //                                             Sequential version                                               //
    //--------------------------------------------------------------------------------------------------------------//
    startTime = MPI_Wtime();
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      std::vector<int> seqHist(gHistSize);

      // Compute histogram
      std::for_each(gArray.begin(), gArray.end(), [&seqHist, hash](int value)
                    { seqHist[hash(value)]++; });

      // Print histogram
      std::printf("Seq histogram: \n");
      printHist(seqHist.data(), gHistSize);
    }

    seqTime = MPI_Wtime() - startTime;
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // We have an array of numbers. The goal is to calculate distributed histogram.
    // The array is first scattered over all ranks. Every rank holds a part of the histogram.
    // We first calculate the histogram locally, then accumulate the parts into the global histogram.
    //   - It is a faster solution, but may not be efficient.
    //   - One sided accumulation can be replaced by MPI_Reduce_scatter).

    /// *** You have to place Fences at several places to prevent data races! ***

    startTime = MPI_Wtime();

    // 1. Distributed the global array (gArray) into lArray. Every rank has lArraySize elements.
    lArray.resize(lArraySize);

    MPI_Scatter(gArray.data(), lArraySize, MPI_INT, lArray.data(), lArraySize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // 2. Declare MPI window for the histogram an MPI_Info object.

    MPI_Win win;
    MPI_Info info;
    MPI_Info_create(&info);

    // 3.  Create an object set same_size and same_disp_unit to true

    MPI_Info_set(info, "same_size", "true");
    MPI_Info_set(info, "same_disp_unit", "true");

    // 4. Allocate the window. Use the MPI_Win_allocate routine to create the window and allocate memory for it).

    MPI_Win_allocate(sizeof(int) * lHistSize, sizeof(int), info, MPI_COMM_WORLD, &distHist, &win);

    // Clear my part of the distributed histogram and local scratch place.
    std::fill_n(distHist, lHistSize, 0);
    std::fill(localHist.begin(), localHist.end(), 0);

    // 6. Calculate local histogram: localHist[hash(lArray[i])]++

    localHist.resize(gHistSize);

    std::for_each(lArray.begin(), lArray.end(), [&localHist, hash](int value)
                  { localHist[hash(value)]++; });

    // 7. Use MPI_Accumulate routine to increment the appropriate bin. Use only one call per rank.

    MPI_Win_fence(MPI_MODE_NOPRECEDE | MPI_MODE_NOPUT | MPI_MODE_NOSTORE, win);

    for (int i = 0; i < gHistSize; i++)
    {

      int rank = i / lHistSize;
      int disp = i % lHistSize;

      // printf("Rank %d, disp %d, rank %d\n", rank, disp, mpiGetCommRank(MPI_COMM_WORLD));

      MPI_Accumulate(&localHist[i], 1, MPI_INT, rank, disp, 1, MPI_INT, MPI_SUM, win);
    }

    MPI_Win_fence(MPI_MODE_NOSUCCEED, win);
    // Print histogram
    std::vector<int> gHist{};

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      gHist.resize(gHistSize);
    }

    // 8. Collect all data from all ranks into a single histogram on the root.

    MPI_Gather(distHist, lHistSize, MPI_INT, gHist.data(), lHistSize, MPI_INT, MPI_ROOT_RANK, MPI_COMM_WORLD);

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      std::printf("Dist histogram: \n");
      printHist(gHist.data(), gHistSize);
    }

    // 9. Free all used variables and MPI objects.

    MPI_Win_free(&win);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    parTime = MPI_Wtime() - startTime;

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiPrintf(MPI_ROOT_RANK, "| Seq time: %6.2fs                                                 |\n",
              seqTime);
    mpiPrintf(MPI_ROOT_RANK, "| Par time: %6.2fs (P = %2d)                                        |\n",
              parTime, mpiGetCommSize(MPI_COMM_WORLD));
    mpiPrintf(MPI_ROOT_RANK, "| Speedup:  %6.2fx (%4.2f%%)                                        |\n",
              seqTime / parTime, (seqTime / parTime) / mpiGetCommSize(MPI_COMM_WORLD) * 100);
    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiFlush();
    break;
  } // case 4

    //--------------------------------------------------------------------------------------------------------------------//
    //                                   Example 5 - Distributed matrix multiplication                                    //
    //--------------------------------------------------------------------------------------------------------------------//
  case 5:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 5 - Distributed matrix multiplications.                     \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Only square matrices are supported. For debugging purposes, use matrix 16x16.
#ifdef DEBUG
    constexpr int nRows = 16;
    constexpr int nCols = nRows;
#else
    constexpr int nRows = 2048;
    constexpr int nCols = nRows;
#endif

    // Matrices distributed over rows.
    const int localRows = nRows / mpiGetCommSize(MPI_COMM_WORLD);
    const int localCols = nCols;

    // C = A * B
    // Three complete matrices, valid only at root.
    std::vector<double> globalA{};
    std::vector<double> globalB{};
    // Result gathered from other ranks at the end.
    std::vector<double> globalC{};
    /// Seq value for comparison.
    std::vector<double> seqC{};

    /// Local parts of matrix A, B and C at each rank [nRows / P, nCols]
    std::vector<double> localA{};
    std::vector<double> localC{};

    double *localB{};

    mpiPrintf(MPI_ROOT_RANK, " Generating matrix [%d x %d] ...\n", nRows, nCols);

    // Root allocates all global matrices and initializes them.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      globalA.resize(nRows * nCols);
      globalB.resize(nRows * nCols);
      globalC.resize(nRows * nCols);
      seqC.resize(nRows * nCols);

      std::random_device dev{};
      std::mt19937 gen{dev()};
      std::uniform_real_distribution<double> dis(-1.0, 1.0);

      std::generate(globalA.begin(), globalA.end(), [&dis, &gen]
                    { return dis(gen); });
      std::generate(globalB.begin(), globalB.end(), [&dis, &gen]
                    { return dis(gen); });
    }

    double startTime{}, seqTime{}, parTime{};

    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//

#ifndef DEBUG
    mpiPrintf(MPI_ROOT_RANK, " Seq version ... \n");
#endif
    // Seq version of Matrix multiplication
    startTime = MPI_Wtime();
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      // Square matrix, nRows == nCols.
      // j and k loops are swapped to enable automatic vectorization, AVS students should know :-)
      for (std::size_t i{}; i < nRows; i++)
      {
        for (std::size_t k{}; k < nRows; k++)
        {
          for (std::size_t j{}; j < nRows; j++)
          {
            seqC[i * nRows + j] += globalA[i * nRows + k] + globalB[k * nRows + j];
          }
        }
      }
    }
    seqTime = MPI_Wtime() - startTime;

    // printf seq result
#ifdef DEBUG
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      printMatrix(seqC, nRows, nCols);
    }
#endif

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
#ifndef DEBUG
    mpiPrintf(MPI_ROOT_RANK, " Par version ... \n");
#endif
    mpiFlush();

    //--------------------------------------------------------------------------------------------------------------//

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Imagine we are about to calculate a distributed matrix - matrix multiplication.
    // Square matrices globalA, globalB and globalC are distributed evenly over all ranks (each rank has a few rows)
    // into localA, localB and localC.
    // Every rank multiplies its localA rows with B columns and store it into localC row.
    //
    // For the multiplication, the localA and localC matrices are local, however, the B matrix is distributed.
    // Thus we will have to collect the columns from other ranks using MPI_Get operation.
    // Of course, we want to minimize the number of Get operations :)

    // *** Be careful and place fences at appropriate places ***
    startTime = MPI_Wtime();

    // 1. Declare a window for globalB matrix.

    MPI_Win winB;

    // 2. Allocate window and memory for B matrix (localB). The matrix has localRows and nCols.

    MPI_Win_allocate(sizeof(double) * localRows * nCols, sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &localB, &winB);

    // 3. Allocate memory for localA and localC.

    localA.resize(localRows * nCols);
    localC.resize(localRows * nCols);

    // 4. Define an rowType datatype to scatter rows over ranks.

    // 5. Define colChunkType datatype. This datatype covers 1 column over the local array. We'll use it for reading
    //    column chunks from other ranks.

    // 6. Scatter matrices A and B, use a nonblocking scatters and overlap them with zeroing localC matrix.

    // Fences before the computation starts.

    // 7. Perform matrix multiplication. To get the best performance, organize the loops in this order [j, i, k].
    //    For every column (j), collect the complete column from all ranks. You have to loop over all ranks and
    //    Get one colChunkType datatype and store it in a local variable oneCol.
    //    Then perform matrix multiplication like this:
    //       localC[i * nCols + j] += localA[i * nCols + k] + oneCol[k];
    //    @warning There must be a fence somewhere :)

    // 8. Gather the localC matrix into the globalC matrix for comparison.

    parTime = MPI_Wtime() - startTime;

    mpiFlush();

    // print par result and maximal error
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
#ifdef DEBUG
      printMatrix(globalC.data(), nRows, nCols);
#endif

      double maxError{};
      for (std::size_t i{}; i < nRows * nCols; i++)
      {
        maxError = std::max(maxError, std::abs(globalC[i] - seqC[i]));
      }
      mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
      mpiPrintf(MPI_ROOT_RANK, " Max error: %e\n", maxError);
    }

    // 9. Delete all allocated MPI objects.

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiPrintf(MPI_ROOT_RANK, "| Seq time: %7.2fs                                                |\n",
              seqTime);
    mpiPrintf(MPI_ROOT_RANK, "| Par time: %7.2fs (P = %2d)                                       |\n",
              parTime, mpiGetCommSize(MPI_COMM_WORLD));
    mpiPrintf(MPI_ROOT_RANK, "| Speedup:  %7.2fx (%6.2f%%)                                      |\n",
              seqTime / parTime, (seqTime / parTime) / mpiGetCommSize(MPI_COMM_WORLD) * 100);
    mpiPrintf(MPI_ROOT_RANK, "+-------------------------------------------------------------------+\n");
    mpiFlush();
    break;
  } // case 5

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
