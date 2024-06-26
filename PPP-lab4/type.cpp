/**
 * @file      type.cpp
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
 * @brief     PC lab 4 / MPI datatypes
 *
 * @version   2024
 *
 * @date      08 March     2020, 16:47 (created) \n
 * @date      14 March     2023, 14:43 (created) \n
 * @date      12 March     2024, 15:59 (revised) \n
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
 * @param argc.
 * @param argv.
 * @return test id.
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
                const int nRows,
                const int nCols)
{
  for (int i = 0; i < nRows; i++)
  {
    for (int j = 0; j < nCols; j++)
    {
      matrix[i * nCols + j] = i * 100 + j;
    }
  }
} // end of initMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Print matrix.
 * @param [in] matrix - Matrix to print.
 * @param [in] nRows  - Number of rows.
 * @param [in] nCols  - Number of cols.
 */
void printMatrix(const int *matrix,
                 int nRows,
                 int nCols)
{
  static constexpr std::size_t maxTmpLen = 16;

  const int rank = mpiGetCommRank(MPI_COMM_WORLD);

  std::string str{};
  char val[maxTmpLen]{};

  for (int i = 0; i < nRows; i++)
  {
    str.clear();

    for (int j = 0; j < nCols; j++)
    {
      std::snprintf(val, maxTmpLen, "%4d%s", matrix[i * nCols + j], (j < nCols - 1) ? ", " : "");
      str += val;
    }

    std::printf(" - Rank %d = [%s]\n", rank, str.c_str());
  }
} // end of printMatrix
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
  mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 4                                \n");
  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiFlush();

  // Parse parameters
  const int testId = parseParameters(argc, argv);

  // Select test
  switch (testId)
  {
    //--------------------------------------------------------------------------------------------------------------------//
    //                       Example 1 - Create a contiguous datatype to scatter rows of the matrix                       //
    //--------------------------------------------------------------------------------------------------------------------//
  case 1:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 1 - Create a contiguous datatype to scatter matrix rows     \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    constexpr int nRows = 16;
    constexpr int nCols = 16;
    const int blockSize = nRows / mpiGetCommSize(MPI_COMM_WORLD);

    std::vector<int> matrix{};
    std::vector<int> block{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Allocate matrix [nRows * nCols] in the root, initialize it using initMatrix and print using printMatrix.

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      matrix.resize(nRows * nCols);
      initMatrix(matrix.data(), nRows, nCols);
      printMatrix(matrix.data(), nRows, nCols);
    }

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 2. Allocate block of rows [blockSize * nCols] at every rank.

    block.resize(blockSize * nCols);

    // 3. Declare, create and commit a contiguous datatype describing one row.
    MPI_Datatype ROW_TYPE;
    MPI_Type_contiguous(nCols, MPI_INT, &ROW_TYPE);
    MPI_Type_commit(&ROW_TYPE);

    // 4. Scatter the matrix rows over the ranks

    MPI_Scatter(matrix.data(), blockSize, ROW_TYPE, block.data(), blockSize, ROW_TYPE, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // 5. Print block of rows at every rank.

    printMatrix(block.data(), blockSize, nCols);

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 6. Free the datatype.

    MPI_Type_free(&ROW_TYPE);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 1

    //--------------------------------------------------------------------------------------------------------------------//
    //                     Example 2 - Createa vector datatype to send column blocks to other ranks                      //
    //--------------------------------------------------------------------------------------------------------------------//
  case 2:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 2 - Create a vector datatype to send column block to others \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    constexpr int nRows = 16;
    constexpr int nCols = 16;
    const int blockSize = nCols / mpiGetCommSize(MPI_COMM_WORLD);

    std::vector<int> matrix{};
    std::vector<int> block{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Allocate matrix in the root and initialize it by calling initMatrix.

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 2. Allocate a block of columns [nRows * blockSize] at every rank.

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      matrix.resize(nRows * nCols);
      initMatrix(matrix.data(), nRows, nCols);
      printMatrix(matrix.data(), nRows, nCols);
    }

    block.resize(nRows * blockSize);

    // 3. Declare two MPI data types, one for the sender (COL_TYPE) and another one (block of cols) for the receivers.

    MPI_Datatype COL_TYPE;
    MPI_Datatype BLOCK_COL_TYPE;

    // 4. In the root, create a vector datatype that holds blocks of the data (blockSize * nRows).

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      MPI_Type_vector(nRows, blockSize, nCols, MPI_INT, &COL_TYPE);
      MPI_Type_commit(&COL_TYPE);
    }
    // 5. In all ranks, create a contiguous datatype to received block of data.

    MPI_Type_contiguous(nRows * blockSize, MPI_INT, &BLOCK_COL_TYPE);
    MPI_Type_commit(&BLOCK_COL_TYPE);

    const int worldSize = mpiGetCommSize(MPI_COMM_WORLD);

    // 6. Send a given number of columns to the other ranks. Don't forget to send and receive data in the root rank.
    //    To prevent deadlock, use non-blocking sends.
    //    Send the whole block to particular sender at once (1 element of a given datatype).

    MPI_Request requests[mpiGetCommSize(MPI_COMM_WORLD)];

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      for (int i = 0; i < worldSize; i++)
      {
        MPI_Isend(matrix.data() + i * blockSize, 1, COL_TYPE, i, 0, MPI_COMM_WORLD, &requests[i]);
      }
    }

    MPI_Recv(block.data(), 1, BLOCK_COL_TYPE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      MPI_Waitall(mpiGetCommSize(MPI_COMM_WORLD), requests, MPI_STATUSES_IGNORE);
    }

    // 7. Print block of rows at every rank.

    printMatrix(block.data(), nRows, blockSize);

    // 8. Free the data types.

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      MPI_Type_free(&COL_TYPE);
    }
    MPI_Type_free(&BLOCK_COL_TYPE);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 2

    //--------------------------------------------------------------------------------------------------------------------//
    //                        Example 3 - Create a vector datatype to scatter columns of the matrix                       //
    //--------------------------------------------------------------------------------------------------------------------//
  case 3:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 3 - Create a vector datatype to scatter column blocks       \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    constexpr int nRows = 16;
    constexpr int nCols = 16;
    const int blockSize = nCols / mpiGetCommSize(MPI_COMM_WORLD);

    std::vector<int> matrix{};
    std::vector<int> block{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Allocate matrix in the root and initialize it by calling initMatrix.

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      matrix.resize(nRows * nCols);
      initMatrix(matrix.data(), nRows, nCols);
      printMatrix(matrix.data(), nRows, nCols);
    }

    // 2. Allocate a block of columns [nRows * blockSize] at every rank.

    block.resize(nRows * blockSize);

    // 3. Declare 2 MPI data types - MAT_COL_TYPE for sender and BLOCK_COL_TYPE for receivers.

    MPI_Datatype MAT_COL_TYPE;
    MPI_Datatype BLOCK_COL_TYPE;

    // 4. In the root rank, create a vector datatype for 1 column.
    //    Create a resized data type to allow sending multiple columns.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      // 4.1 Define the original vector type variable

      MPI_Datatype ORIGINAL_VECTOR_TYPE;

      // 4.2 Create the original vector type

      MPI_Type_vector(nRows, 1, nCols, MPI_INT, &ORIGINAL_VECTOR_TYPE);

      // 4.3 Create the resized vector type from the original one

      MPI_Type_create_resized(ORIGINAL_VECTOR_TYPE, 0, sizeof(int), &MAT_COL_TYPE);

      // 4.4 Commit the resized vector type

      MPI_Type_commit(&MAT_COL_TYPE);

      // 4.5 Free the original vector type

      MPI_Type_free(&ORIGINAL_VECTOR_TYPE);
    }

    // 5. In all ranks, create a vector datatype to receive one column.
    //    If the blockSize == 1, duplicate this datatype to the resized one.
    //    else create a resized datatype
    {
      // 5.1 Define the original vector type variable

      MPI_Datatype ORIGINAL_VECTOR_TYPE;

      // 5.2 Create the original vector type

      MPI_Type_vector(nRows, 1, blockSize, MPI_INT, &ORIGINAL_VECTOR_TYPE);

      if (blockSize > 1)
      {
        // 5.3a Create the resized vector type from the original one

        MPI_Type_create_resized(ORIGINAL_VECTOR_TYPE, 0, sizeof(int), &BLOCK_COL_TYPE);

        // 5.3b Free the original vector type

        MPI_Type_free(&ORIGINAL_VECTOR_TYPE);
      }
      else
      {
        // 5.3c Move the original vector type to the resized one

        BLOCK_COL_TYPE = ORIGINAL_VECTOR_TYPE;
      }

      // 5.4 Commit the resized vector type
      MPI_Type_commit(&BLOCK_COL_TYPE);
    }

    // 6. Scatter the matrix rows over the ranks. Every rank receives blockSize columns.
    //    Use the resized data types.

    MPI_Scatter(matrix.data(), blockSize, MAT_COL_TYPE, block.data(), blockSize, BLOCK_COL_TYPE, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // 7. Print block of rows at every rank.

    printMatrix(block.data(), blockSize, nRows);

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 8. Free datatype at root and all ranks.

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      MPI_Type_free(&MAT_COL_TYPE);
    }
    MPI_Type_free(&BLOCK_COL_TYPE);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 3

    //--------------------------------------------------------------------------------------------------------------------//
    //                        Example 4 - Create a subarray datatype to scatter matrix by tiles                           //
    //--------------------------------------------------------------------------------------------------------------------//
  case 4:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 4 - Create a subarray datatype to scatter matrix by tiles   \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    constexpr int nRows = 16;
    constexpr int nCols = 16;
    const int tileSize = nCols / sqrt(mpiGetCommSize(MPI_COMM_WORLD));

    std::vector<int> matrix{};
    std::vector<int> tile{};

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                            Enter your code here                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Allocate matrix in the root and initialize it by calling initMatrix.

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      matrix.resize(nRows * nCols);
      initMatrix(matrix.data(), nRows, nCols);
      printMatrix(matrix.data(), nRows, nCols);
    }

    // 2. Allocate a square tile [tileSize * tileSize] at every rank.

    tile.resize(tileSize * tileSize);

    // 3. Declare 2 MPI data types - tileType to scatter the tile and blockType in each receiver to receive a
    //    MPI_Type_contiguous tile.

    MPI_Datatype TILE_TYPE;
    MPI_Datatype BLOCK_TYPE;

    // 4. In the root rank, create a tile datatype and its resized version
    //    Create a resized data type to allow sending multiple columns.

    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      // 4.1 Define the original subarray type variable

      MPI_Datatype ORIGINAL_SUBARRAY_TYPE;

      // 4.2 Create the original subarray type

      int sizes[2] = {nRows, nCols};
      int subsizes[2] = {tileSize, tileSize};
      int starts[2] = {0, 0};

      MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &ORIGINAL_SUBARRAY_TYPE);

      // 4.3 Create the resized subarray type from the original one

      MPI_Type_create_resized(ORIGINAL_SUBARRAY_TYPE, 0, sizeof(int), &TILE_TYPE);

      // 4.4 Commit the resized subarray type

      MPI_Type_commit(&TILE_TYPE);

      // 4.5 Free the original subarray type

      MPI_Type_free(&ORIGINAL_SUBARRAY_TYPE);
    }

    // 5. In all ranks, create a contiguous datatype to receive one tile

    MPI_Type_contiguous(tileSize * tileSize, MPI_INT, &BLOCK_TYPE);
    MPI_Type_commit(&BLOCK_TYPE);

    // 6. Calculate counts and displacements for Scatterv

    std::vector<int> sendCounts(mpiGetCommSize(MPI_COMM_WORLD));
    std::vector<int> displacements(mpiGetCommSize(MPI_COMM_WORLD));

    for (int i = 0; i < mpiGetCommSize(MPI_COMM_WORLD); i++)
    {

      sendCounts[i] = 1;
      displacements[i] = ((i * tileSize) / nCols) * nRows * tileSize + (i * tileSize) % nCols;
    }
    // 7. Scatter the matrix tiles over the ranks

    MPI_Scatterv(matrix.data(), sendCounts.data(), displacements.data(), TILE_TYPE, tile.data(), 1, BLOCK_TYPE, MPI_ROOT_RANK, MPI_COMM_WORLD);

    // 8. Print block of rows at every rank.

    printMatrix(tile.data(), tileSize, tileSize);

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 8. Free datatype at root and all ranks.

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 4

  default:
  {
    mpiPrintf(MPI_ROOT_RANK, " !!!                     Unknown test number                      !!!\n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    break;
  }
  } // switch

  MPI_Finalize();
} // end of main
//----------------------------------------------------------------------------------------------------------------------
