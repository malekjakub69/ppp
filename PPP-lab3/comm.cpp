/**
 * @file      comm_solution.cpp
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
 * @brief     PC lab 3 / MPI Communicators
 *
 * @version   2024
 *
 * @date      29 February  2020, 09:04 (created) \n
 * @date      07 March     2023, 18:29 (revised) \n
 * @date      05 March     2024, 15:52 (revised) \n
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
//------------------------------------------------------------------------------------------------------------------------

/**
 * Get MPI rank within the communicator.
 * @param [in] comm - actual communicator.
 * @return rank within the comm.
 */
int mpiGetCommRank(MPI_Comm comm)
{
  int rank{MPI_UNDEFINED};

  MPI_Comm_rank(comm, &rank);

  return rank;
} // end of mpiGetCommRank
//------------------------------------------------------------------------------------------------------------------------

/**
 * Get size of the communicator.
 * @param [in] comm - actual communicator.
 * @return number of ranks within the comm.
 */
int mpiGetCommSize(MPI_Comm comm)
{
  int size{-1};

  MPI_Comm_size(comm, &size);

  return size;
} // end of mpiGetCommSize
//------------------------------------------------------------------------------------------------------------------------

/**
 * Routine that handles MPI errors in PPP communicator.
 * @param [in, out] comm - Communicator where the error occurred.
 * @param [in, out] err  - Integer holding the error
 * @param [in] ...       - A variable number of other parameters.
 */
void pppErrorHandler(MPI_Comm *comm, int *err, ...)
{
  // List of error codes can be found here: https://linux.die.net/man/3/openmpi

  std::array<char, MPI_MAX_OBJECT_NAME> commName{};
  int commNameSize{};

  // Get name of given communicator
  MPI_Comm_get_name(*comm, commName.data(), &commNameSize);

  // Print error message
  mpiPrintf(MPI_ALL_RANKS, " !!! Rank %d found an error %d in communicator %s !!! \n",
            mpiGetCommRank(*comm), *err, commName.data());

  // Call MPI_Abort if you don't know what to next.
  // MPI_Abort(*comm, *err);
} // end of pppErrorHandler
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
  mpiPrintf(MPI_ROOT_RANK, "                            PPP Lab 3                                \n");
  mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
  mpiFlush();

  // Parse parameters
  const int testId = parseParameters(argc, argv);

  // Select test
  switch (testId)
  {
    //--------------------------------------------------------------------------------------------------------------------//
    //                                Example 1 - Duplicate communicator and give it a name                               //
    //--------------------------------------------------------------------------------------------------------------------//
  case 1:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 1 - Duplicate communicator and give it a name               \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Declare a new communicator.

    MPI_Comm pppComm;

    // 2. Duplicate communicator.

    MPI_Comm_dup(MPI_COMM_WORLD, &pppComm);

    // 3. Set communicator name.

    MPI_Comm_set_name(pppComm, "PPP_Communicator");

    // 4. Rank 3 does something bad - it sends a message to rank 100 which does not exist.

    if (mpiGetCommRank(pppComm) == 3)
    {
      MPI_Send(nullptr, 0, MPI_INT, 100, 0, pppComm);
    }

    // 5. Free communicator

    MPI_Comm_free(&pppComm);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 1

    //--------------------------------------------------------------------------------------------------------------------//
    //                               Example 2 - Duplicate communicator set error handler.                                //
    //--------------------------------------------------------------------------------------------------------------------//
  case 2:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 2 - Duplicate communicator set error handler                \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Declare a new communicator.

    MPI_Comm pppComm;

    // 2. Duplicate communicator.

    MPI_Comm_dup(MPI_COMM_WORLD, &pppComm);

    // 3. Set communicator name.

    MPI_Comm_set_name(pppComm, "PPP_Communicator");

    // 4. Define MPI error handler.

    MPI_Errhandler pppError;

    // 5. Create error handler - link function pointer to pppErrorHandler(...) to Handler.

    MPI_Comm_create_errhandler(pppErrorHandler, &pppError);

    // 6. Set error handler to the PPP communicator.

    MPI_Comm_set_errhandler(pppComm, pppError);

    // 7. Rank 3 does something bad - it sends a message to rank 100 which does not exist.

    if (mpiGetCommRank(pppComm) == 3)
    {
      MPI_Send(nullptr, 0, MPI_INT, 100, 0, pppComm);
    }

    // 8. Free MPI  Error handler.

    MPI_Errhandler_free(&pppError);

    // 9. Free communicator

    MPI_Comm_free(&pppComm);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 2

    //--------------------------------------------------------------------------------------------------------------------//
    //                               Example 3 - Split MPI_COMM_WORLD into row and col communicators.                     //
    //--------------------------------------------------------------------------------------------------------------------//
  case 3:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 3 - Split MPI_COMM_WORLD into row and col communicators     \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    const int rowSize = static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));
    const int colSize = static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Declare a row and col communicator.

    MPI_Comm rowComm;
    MPI_Comm colComm;

    // 2. Create a row communicator using split.

    MPI_Comm_split(MPI_COMM_WORLD, mpiGetCommRank(MPI_COMM_WORLD) / rowSize, mpiGetCommRank(MPI_COMM_WORLD) % rowSize, &rowComm);

    // 3. Create a col communicator using split.
    MPI_Comm_split(MPI_COMM_WORLD, mpiGetCommRank(MPI_COMM_WORLD) % colSize, mpiGetCommRank(MPI_COMM_WORLD) / colSize, &colComm);

    // 4. Print your rank in the new and old communicators.

    printf(" Row comm: %2d /%2d, Col comm: %2d / %2d,  COMM_WORLD: %2d/%2d \n", mpiGetCommRank(rowComm), mpiGetCommSize(rowComm),
           mpiGetCommRank(colComm), mpiGetCommSize(colComm), mpiGetCommRank(MPI_COMM_WORLD), mpiGetCommSize(MPI_COMM_WORLD));

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // 5. Root ranks of each row and col communicator set the row and coll ID.

    int rowRootId, colRootId;

    if (mpiGetCommRank(rowComm) == MPI_ROOT_RANK)
    {
      rowRootId = mpiGetCommRank(MPI_COMM_WORLD) / rowSize;
    }
    if (mpiGetCommRank(colComm) == MPI_ROOT_RANK)
    {
      colRootId = mpiGetCommRank(MPI_COMM_WORLD) % colSize;
    }

    // 6. Broadcast row and col ID within row and col communicator.

    MPI_Bcast(&rowRootId, 1, MPI_INT, MPI_ROOT_RANK, rowComm);
    MPI_Bcast(&colRootId, 1, MPI_INT, MPI_ROOT_RANK, colComm);

    // 7. Print out row and col id in all processes.

    printf("Row and Col Id [%d, %d], World rank %d \n", rowRootId, colRootId, mpiGetCommRank(MPI_COMM_WORLD));

    // 8. Free communicators

    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  } // case 3

    //--------------------------------------------------------------------------------------------------------------------//
    //                               Example 4 - Divide communicator into 2x2 square tiles.                               //
    //--------------------------------------------------------------------------------------------------------------------//
  case 4:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 4 - Divide communicator into 2x2 square tiles               \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    const int rowSize = static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));
    const int colSize = static_cast<int>(std::sqrt(mpiGetCommSize(MPI_COMM_WORLD)));

    const int tileSize = 2;
    const int tileCount = rowSize / tileSize;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // 1. Lambda getting the tile ID of a given rank
    auto getTileId = [rowSize, colSize, tileSize, tileCount](int rank) -> int
    {
      return (rank / rowSize) / tileSize * tileCount + (rank % colSize) / tileSize;
    };

    // 2. Declare a tile communicator.

    MPI_Comm tileComm;

    // 3. Declare two groups, one that contains all ranks from MPI_COMM_WORLD and the other one that will only contain
    //    ranks in this tile.

    MPI_Group worldGroup;
    MPI_Group tileGroup;

    // 4. Get the group out of the MPI_COMM_WORLD communicator.

    MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);

    // 5. Create a vector of ranks that belongs to current tile.
    std::vector<int> tileRanks{};
    tileRanks.reserve(tileSize * tileSize);

    // 6. Add all ranks in the same tile as I belong to in tileRanks

    for (int i = 0; i < mpiGetCommSize(MPI_COMM_WORLD); i++)
    {
      if (getTileId(i) == getTileId(mpiGetCommRank(MPI_COMM_WORLD)))
      {
        tileRanks.push_back(i);
      }
    }

    // 7. Include tileRanks into the MPI group.

    MPI_Group_incl(worldGroup, tileRanks.size(), tileRanks.data(), &tileGroup);

    // 8. Create a new communicator based on the MPI group.

    MPI_Comm_create(MPI_COMM_WORLD, tileGroup, &tileComm);

    // 9. Each root of the tile sends its original rank to the tiles.

    int myRootRank;

    if (mpiGetCommRank(tileComm) == MPI_ROOT_RANK)
    {
      myRootRank = mpiGetCommRank(MPI_COMM_WORLD);
    }

    MPI_Bcast(&myRootRank, 1, MPI_INT, MPI_ROOT_RANK, tileComm);

    // 10. Print my original rank, my new rank, my leader's rank in the MPI_COMM_WORLD
    printf("Who am I? [original rank, new rank, leader] = [%2d, %2d, %2d]\n", mpiGetCommRank(MPI_COMM_WORLD), mpiGetCommRank(tileComm), myRootRank);
    // 11. Free the groups and communicators.

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  }

    //--------------------------------------------------------------------------------------------------------------------//
    //                               Example 5 - Dual layer decomposition of vector dot-product.                          //
    //--------------------------------------------------------------------------------------------------------------------//
  case 5:
  {
    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, " Example 5 - Dual layer decomposition of vector dot-product          \n");
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    // Global array and global result
    std::vector<int> a{};
    std::vector<int> b{};

    constexpr std::size_t size = 1024 * 1024; // 1M elements, 16 cores, 4MB in total

    int parResult{};
    int seqResult{};

    // Initialize arrays and calculate a sequential version.
    if (mpiGetCommRank(MPI_COMM_WORLD) == MPI_ROOT_RANK)
    {
      a.resize(size);
      b.resize(size);

      std::random_device rd{};
      std::mt19937 gen{rd()};

      std::uniform_int_distribution<int> disA{0, 100};
      std::uniform_int_distribution<int> disB{0, 20};

      // Init array
      std::generate(std::begin(a), std::end(a), [&disA, &gen]()
                    { return disA(gen); });
      std::generate(std::begin(b), std::end(b), [&disB, &gen]()
                    { return disB(gen); });

      seqResult = std::transform_reduce(std::begin(a), std::end(a), std::begin(b), 0);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                        Enter your code here                                                  //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int level2Result{};
    int level1Result{};

    // 1. Declare communicator for the first (MPI_COMM_Level1) and the second layer decomposition (MPI_COMM_Level2)
    // Lvl 1 :                        [0  4  8  12]
    // LvL 2 :    [0  1  2  3]  [4  5  6  7]  [8  9  10  11] [12  13  14  15]

    MPI_Comm MPI_COMM_Level1;
    MPI_Comm MPI_COMM_Level2;

    // 2. Define array sizes at level 1 and level 2

    const int level1Size = 4;
    const int level2Size = 4;

    // 3. Create communicators for level 1 and level 2.

    // Level 1
    if (mpiGetCommRank(MPI_COMM_WORLD) % level1Size == 0)
    {
      MPI_Comm_split(MPI_COMM_WORLD, mpiGetCommRank(MPI_COMM_WORLD) % level1Size, 0, &MPI_COMM_Level1);
    }
    else
    {
      MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, 0, &MPI_COMM_Level1);
    }

    // 6. Set error handler to the PPP communicator.

    // Level 2
    MPI_Comm_split(MPI_COMM_WORLD, mpiGetCommRank(MPI_COMM_WORLD) / level2Size, mpiGetCommRank(MPI_COMM_WORLD) % level2Size, &MPI_COMM_Level2);

    // 4. Allocate local arrays for level 1 and for level 2. For level 1 arrays, allocate only at those ranks.
    std::vector<int> level1A{};
    std::vector<int> level1B{};

    int sendSizeLevel1 = size / level1Size;

    if (MPI_COMM_Level1 != MPI_COMM_NULL)
    {
      level1A.resize(sendSizeLevel1);
      level1B.resize(sendSizeLevel1);

      // 5. Scatter vector a and b over level 1 group. Test whether the communicator is not equal to MPI_COMM_NULL.

      MPI_Scatter(a.data(), sendSizeLevel1, MPI_INT, level1A.data(), sendSizeLevel1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_Level1);
      MPI_Scatter(b.data(), sendSizeLevel1, MPI_INT, level1B.data(), sendSizeLevel1, MPI_INT, MPI_ROOT_RANK, MPI_COMM_Level1);
    }

    // 6. Scatter vector a and b over level 2 group.

    std::vector<int> level2A{};
    std::vector<int> level2B{};

    int sendSizeLevel2 = size / (level1Size * level2Size);

    level2A.resize(sendSizeLevel2);
    level2B.resize(sendSizeLevel2);

    MPI_Scatter(level1A.data(), sendSizeLevel2, MPI_INT, level2A.data(), sendSizeLevel2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_Level2);
    MPI_Scatter(level1B.data(), sendSizeLevel2, MPI_INT, level2B.data(), sendSizeLevel2, MPI_INT, MPI_ROOT_RANK, MPI_COMM_Level2);

    // 7. Calculate local dot product.

    int localResult = 0;
    for (int i = 0; i < sendSizeLevel2; i++)
    {
      localResult += level2A[i] * level2B[i];
    }

    // 8. Reduce partial results at level 2.

    MPI_Reduce(&localResult, &level2Result, 1, MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_Level2);

    // 9. Reduce partial at level 1. Do it only at ranks where MPI_COMM_Level1 != MPI_COMM_NULL.

    if (MPI_COMM_Level1 != MPI_COMM_NULL)
    {
      MPI_Reduce(&level2Result, &level1Result, 1, MPI_INT, MPI_SUM, MPI_ROOT_RANK, MPI_COMM_Level1);
    }

    if (MPI_ROOT_RANK == mpiGetCommRank(MPI_COMM_WORLD))
    {
      parResult = level1Result;
    }

    // 10. Free used communicators.

    if (MPI_COMM_Level1 != MPI_COMM_NULL)
    {
      MPI_Comm_free(&MPI_COMM_Level1);
    }

    MPI_Comm_free(&MPI_COMM_Level2);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Print results.
    mpiPrintf(MPI_ROOT_RANK, " Seq result = %d\n", seqResult);
    mpiPrintf(MPI_ROOT_RANK, " Par result = %d\n", parResult);
    mpiPrintf(MPI_ROOT_RANK, " Status = %s\n", ((seqResult - parResult) == 0) ? "Ok" : "Fail");

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    mpiFlush();
    mpiPrintf(MPI_ROOT_RANK, "---------------------------------------------------------------------\n");
    mpiFlush();
    break;
  }

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
