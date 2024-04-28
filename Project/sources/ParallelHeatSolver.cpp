/**
 * @file    ParallelHeatSolver.cpp
 *
 * @author  Jakub Málek <xmalek17@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <ios>
#include <string_view>

#include "ParallelHeatSolver.hpp"
#include <thread>

#define DEBUG 0
#define PRINT_RANK 2

void mpiFlush()
{
  std::fflush(stdout);
  // A known hack to correctly order writes
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

int mpiGetCommSize(MPI_Comm comm)
{
  int size{};

  MPI_Comm_size(comm, &size);

  return size;
}

int mpiGetCommRank(MPI_Comm comm)
{
  int rank{};

  MPI_Comm_rank(comm, &rank);

  return rank;
}

int getTileId(int rowSize, int colSize, int tileSize, int tileCount, int rank)
{
  return (rank / rowSize) / tileSize * tileCount + (rank % colSize) / tileSize;
};

void printEcho(std::string message)
{
  printf("Rank %d: %s \n", mpiGetCommRank(MPI_COMM_WORLD), message.c_str());
  mpiFlush();
}

void printMatrix(const float *matrix,
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
      std::snprintf(val, maxTmpLen, "%8.3f%s", matrix[i * nCols + j], (j < nCols - 1) ? ", " : "");
      str += val;
    }

    std::printf(" - Rank %d = [%s]\n", rank, str.c_str());
  }
} // end of printMatrix

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
      std::snprintf(val, maxTmpLen, "%d%s", matrix[i * nCols + j], (j < nCols - 1) ? ", " : "");
      str += val;
    }

    std::printf(" - Rank %d = [%s]\n", rank, str.c_str());
  }
}

ParallelHeatSolver::ParallelHeatSolver(const SimulationProperties &simulationProps,
                                       const MaterialProperties &materialProps)
    : HeatSolverBase(simulationProps, materialProps)
{
  MPI_Comm_size(MPI_COMM_WORLD, &mWorldSize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mWorldRank);
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("ParallelHeatSolver constructor");
  }
  initGridTopology();
  initDataDistribution();
  allocLocalTiles();
  initHaloExchange();
  /**********************************************************************************************************************/
  /*                                  Call init* and alloc* methods in correct order                                    */
  /**********************************************************************************************************************/

  if (!mSimulationProps.getOutputFileName().empty())
  {

    if (mSimulationProps.useParallelIO())
    {
      openOutputFileParallel();
    }
    else if (mpiGetCommRank(cart_comm) == MPI_ROOT_RANK)
    {
      openOutputFileSequential();
    }
    /**********************************************************************************************************************/
    /*                               Open output file if output file name was specified.                                  */
    /*  If mSimulationProps.useParallelIO() flag is set to true, open output file for parallel access, otherwise open it  */
    /*                         only on MASTER rank using sequetial IO. Use openOutputFile* methods.                       */
    /**********************************************************************************************************************/
  }
}

ParallelHeatSolver::~ParallelHeatSolver()
{
  deinitGridTopology();
  deinitDataDistribution();
  deallocLocalTiles();
  deinitHaloExchange();
  /**********************************************************************************************************************/
  /*                                  Call deinit* and dealloc* methods in correct order                                */
  /*                                             (should be in reverse order)                                           */
  /**********************************************************************************************************************/
}

std::string_view ParallelHeatSolver::getCodeType() const
{
  return codeType;
}

void ParallelHeatSolver::initGridTopology()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("initGridTopology");
  }
  // získání globálních rozměrů mřížky
  mSimulationProps.getDecompGrid(globalRows, globalCols);

  // získání velikosti hrany celé mřížky
  edgeSize = mMaterialProps.getEdgeSize();

  // get airflow rate and cooler temperature
  airflowRate = mSimulationProps.getAirflowRate();
  coolerTemp = mMaterialProps.getCoolerTemperature();

  // výpočet lokálních rozměrů mřížky
  localCols = edgeSize / globalCols;
  localRows = edgeSize / globalRows;

  // vytvoření 2D mřížky
  int dims[2] = {globalRows, globalCols};
  int periods[2] = {0, 0};
  int reorder = 1;

  // vytovření komunikátoru pro 2D mřížku
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

  // získání sousedů
  MPI_Cart_shift(cart_comm, 1, 1, &neighbours[LEFT], &neighbours[RIGHT]);
  MPI_Cart_shift(cart_comm, 0, 1, &neighbours[UP], &neighbours[DOWN]);

  // printf("rank: %d, left: %d, right: %d, up: %d, down: %d \n", mpiGetCommRank(cart_comm), neighbours[LEFT], neighbours[RIGHT], neighbours[UP], neighbours[DOWN]);

  // získání souřadnic postředního sloupce
  int coords[2];
  MPI_Cart_coords(cart_comm, mpiGetCommRank(cart_comm), 2, coords);

  // Střední sloupec pro 2D mřížku
  int middle_col = dims[0] / 2;
  int color = (coords[1] == middle_col) ? 1 : MPI_UNDEFINED;

  // printf("rank: %d, col: %d \n", mpiGetCommRank(cart_comm), coords[1]);

  // vytvoření komunikátoru pro střední sloupec
  MPI_Comm_split(cart_comm, color, coords[1], &middle_col_comm);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end initGridTopology");
  }
  /**********************************************************************************************************************/
  /*                          Initialize 2D grid topology using non-periodic MPI Cartesian topology.                    */
  /*                       Also create a communicator for middle column average temperature computation.                */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::deinitGridTopology()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("deinitGridTopology");
  }

  MPI_Comm_free(&cart_comm);

  if (middle_col_comm != MPI_COMM_NULL)
  {
    MPI_Comm_free(&middle_col_comm);
  }

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end deinitGridTopology");
  }
  /**********************************************************************************************************************/
  /*      Deinitialize 2D grid topology and the middle column average temperature computation communicator              */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::initDataDistribution()

{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("initDataDistribution");
  }

  // výpočet počátečních pozic halo zóny unitř mřížky
  startHaloUpInside = haloZoneSize * (localCols + 2 * haloZoneSize) + haloZoneSize;
  startHaloDownInside = (localCols + 2 * haloZoneSize) * localRows + haloZoneSize;
  startHaloLeftInside = haloZoneSize * (localCols + 2 * haloZoneSize) + haloZoneSize;
  startHaloRightInside = haloZoneSize * (localCols + 2 * haloZoneSize) + localCols;

  // výpočet počátečních pozic halo zóny mimo mřížku
  startHaloUpOutside = haloZoneSize;
  startHaloDownOutside = (localCols + 2 * haloZoneSize) * (localRows + haloZoneSize) + haloZoneSize;
  startHaloLeftOutside = haloZoneSize * (localCols + 2 * haloZoneSize);
  startHaloRightOutside = haloZoneSize * (localCols + 2 * haloZoneSize) + localCols + haloZoneSize;

  // vytvoření nového typu pro subarray
  MPI_Datatype ORIGINAL_SUBARRAY_TYPE;
  int sizes[2];
  int subsizes[2];
  int starts[2];

  // --------------------------- TILE_TYPE ---------------------------

  starts[0] = 0;
  starts[1] = 0;

  sizes[0] = edgeSize;
  sizes[1] = edgeSize;

  subsizes[0] = int(edgeSize / globalRows);
  subsizes[1] = int(edgeSize / globalCols);

  // --------------------------- TILE_TYPE_HALO ---------------------------

  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &ORIGINAL_SUBARRAY_TYPE);
  MPI_Type_create_resized(ORIGINAL_SUBARRAY_TYPE, 0, 1 * sizeof(int), &TILE_TYPE_INT);
  MPI_Type_commit(&TILE_TYPE_INT);
  MPI_Type_free(&ORIGINAL_SUBARRAY_TYPE);

  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &ORIGINAL_SUBARRAY_TYPE);
  MPI_Type_create_resized(ORIGINAL_SUBARRAY_TYPE, 0, 1 * sizeof(float), &TILE_TYPE_FLOAT);
  MPI_Type_commit(&TILE_TYPE_FLOAT);
  MPI_Type_free(&ORIGINAL_SUBARRAY_TYPE);

  starts[0] = haloZoneSize;
  starts[1] = haloZoneSize;

  sizes[0] = subsizes[0] + 2 * haloZoneSize;
  sizes[1] = subsizes[1] + 2 * haloZoneSize;

  subsizes[0] = subsizes[0];
  subsizes[1] = subsizes[1];

  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &ORIGINAL_SUBARRAY_TYPE);
  MPI_Type_create_resized(ORIGINAL_SUBARRAY_TYPE, 0, 1 * sizeof(int), &TILE_TYPE_HALO_INT);
  MPI_Type_commit(&TILE_TYPE_HALO_INT);
  MPI_Type_free(&ORIGINAL_SUBARRAY_TYPE);

  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_FLOAT, &ORIGINAL_SUBARRAY_TYPE);
  MPI_Type_create_resized(ORIGINAL_SUBARRAY_TYPE, 0, 1 * sizeof(float), &TILE_TYPE_HALO_FLOAT);
  MPI_Type_commit(&TILE_TYPE_HALO_FLOAT);
  MPI_Type_free(&ORIGINAL_SUBARRAY_TYPE);

  // --------------------------- BLOCK_TYPE ---------------------------
  // vytvoření nového typu pro celý blok dat
  MPI_Type_contiguous(edgeSize * edgeSize, MPI_INT, &BLOCK_TYPE_INT);
  MPI_Type_commit(&BLOCK_TYPE_INT);

  MPI_Type_contiguous(edgeSize * edgeSize, MPI_FLOAT, &BLOCK_TYPE_FLOAT);
  MPI_Type_commit(&BLOCK_TYPE_FLOAT);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end initDataDistribution");
  }

  /**********************************************************************************************************************/
  /*                 Initialize variables and MPI datatypes for data distribution (float and int).                      */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::deinitDataDistribution()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("deinitDataDistribution");
  }
  MPI_Type_free(&TILE_TYPE_INT);
  MPI_Type_free(&TILE_TYPE_FLOAT);

  MPI_Type_free(&TILE_TYPE_HALO_INT);
  MPI_Type_free(&TILE_TYPE_HALO_FLOAT);

  MPI_Type_free(&BLOCK_TYPE_INT);
  MPI_Type_free(&BLOCK_TYPE_FLOAT);
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end deinitDataDistribution");
  }
  /**********************************************************************************************************************/
  /*                       Deinitialize variables and MPI datatypes for data distribution.                              */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::allocLocalTiles()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("allocLocalTiles");
  }

  size_with_halo = (localCols + 2 * haloZoneSize) * (localRows + 2 * haloZoneSize);

  localDomainMap.resize(size_with_halo, 0);
  localDomainParams.resize(size_with_halo, 0.f);

  MPI_Alloc_mem(size_with_halo * sizeof(float), MPI_INFO_NULL, &localTemp[OLD_TEMP]);
  MPI_Alloc_mem(size_with_halo * sizeof(float), MPI_INFO_NULL, &localTemp[NEW_TEMP]);

  std::fill(localTemp[OLD_TEMP], localTemp[OLD_TEMP] + size_with_halo, 0.0f);
  std::fill(localTemp[NEW_TEMP], localTemp[NEW_TEMP] + size_with_halo, 0.0f);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end allocLocalTiles");
  }
  /**********************************************************************************************************************/
  /*            Allocate local tiles for domain map (1x), domain parameters (1x) and domain temperature (2x).           */
  /*                                               Use AlignedAllocator.                                                */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::deallocLocalTiles()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("deallocLocalTiles");
  }

  localDomainMap.clear();
  localDomainParams.clear();

  MPI_Free_mem(localTemp[OLD_TEMP]);
  MPI_Free_mem(localTemp[NEW_TEMP]);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end deallocLocalTiles");
  }
  /**********************************************************************************************************************/
  /*                                   Deallocate local tiles (may be empty).                                           */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::initHaloExchange()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("initHaloExchange");
  }

  // --------------------------- ROW AND COL TYPE ---------------------------

  int count = 2;
  int blenght = localCols;
  int stride = localCols + 2 * haloZoneSize;

  // MPI_Type_vector(count, blenght, stride, MPI_INT, &ROW_TYPE_INT);
  // MPI_Type_commit(&ROW_TYPE_INT);

  MPI_Type_vector(count, blenght, stride, MPI_FLOAT, &ROW_TYPE_FLOAT);
  MPI_Type_commit(&ROW_TYPE_FLOAT);

  count = localRows;
  blenght = 2;
  stride = localCols + 2 * haloZoneSize;

  // MPI_Type_vector(count, blenght, stride, MPI_INT, &COL_TYPE_INT);
  // MPI_Type_commit(&COL_TYPE_INT);

  MPI_Type_vector(count, blenght, stride, MPI_FLOAT, &COL_TYPE_FLOAT);
  MPI_Type_commit(&COL_TYPE_FLOAT);

  if (mSimulationProps.isRunParallelRMA())
  {
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "no_locks", "true");
    MPI_Info_set(info, "same_size", "true");
    MPI_Info_set(info, "same_disp_unit", "true");

    MPI_Win_create(localTemp[OLD_TEMP], (localCols + 2 * haloZoneSize) * (localRows + 2 * haloZoneSize) * sizeof(float),
                   sizeof(float), info, cart_comm, &HALO_WINDOW[0]);

    MPI_Win_create(localTemp[NEW_TEMP], (localCols + 2 * haloZoneSize) * (localRows + 2 * haloZoneSize) * sizeof(float),
                   sizeof(float), info, cart_comm, &HALO_WINDOW[1]);

    MPI_Info_free(&info);
  }

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end initHaloExchange");
  }
  /**********************************************************************************************************************/
  /*                            Initialize variables and MPI datatypes for halo exchange.                               */
  /*                    If mSimulationProps.isRunParallelRMA() flag is set to true, create RMA windows.                 */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::deinitHaloExchange()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("deinitHaloExchange");
  }

  // MPI_Type_free(&ROW_TYPE_INT);
  // MPI_Type_free(&COL_TYPE_INT);

  MPI_Type_free(&ROW_TYPE_FLOAT);
  MPI_Type_free(&COL_TYPE_FLOAT);

  if (mSimulationProps.isRunParallelRMA())
  {
    MPI_Win_free(&HALO_WINDOW[0]);
    MPI_Win_free(&HALO_WINDOW[1]);
  }
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end deinitHaloExchange");
  }
  /**********************************************************************************************************************/
  /*                            Deinitialize variables and MPI datatypes for halo exchange.                             */
  /**********************************************************************************************************************/
}

template <typename T>
void ParallelHeatSolver::scatterTiles(const T *globalData, T *localData)
{

  // TODO: nefunguje pro 1D dekompozici
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("scatterTiles");
  }
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported scatter datatype!");

  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? TILE_TYPE_INT : TILE_TYPE_FLOAT;
  const MPI_Datatype local_tile_type = std::is_same_v<T, int> ? TILE_TYPE_HALO_INT : TILE_TYPE_HALO_FLOAT;

  std::vector<int> displacements(mWorldSize);
  std::vector<int> counts(mWorldSize);

  for (int i = 0; i < mWorldSize; ++i)
  {
    int coords[2];
    MPI_Cart_coords(cart_comm, i, 2, coords);
    counts[i] = 1;
    displacements[i] = coords[0] * localRows * (localCols + 2 * haloZoneSize) + coords[1] * localCols;
  }

  MPI_Scatterv(globalData, counts.data(), displacements.data(), global_tile_type, localData, 1, local_tile_type, 0, cart_comm);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end scatterTiles");
  }
  /**********************************************************************************************************************/
  /*                      Implement master's global tile scatter to each rank's local tile.                             */
  /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
  /*                                                                                                                    */
  /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
  /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
  /**********************************************************************************************************************/
}

template <typename T>
void ParallelHeatSolver::gatherTiles(const T *localData, T *globalData)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("gatherTiles");
  }
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, float>, "Unsupported gather datatype!");

  const MPI_Datatype local_tile_type = std::is_same_v<T, int> ? TILE_TYPE_HALO_INT : TILE_TYPE_HALO_FLOAT;
  const MPI_Datatype global_tile_type = std::is_same_v<T, int> ? TILE_TYPE_INT : TILE_TYPE_FLOAT;

  std::vector<int> displacements(mWorldSize);
  std::vector<int> counts(mWorldSize);

  for (int i = 0; i < mWorldSize; ++i)
  {
    int coords[2];
    MPI_Cart_coords(cart_comm, i, 2, coords);
    counts[i] = 1;
    displacements[i] = coords[0] * localRows * edgeSize + coords[1] * localCols;
  }

  MPI_Gatherv(localData, 1, local_tile_type, globalData, counts.data(), displacements.data(), global_tile_type, MPI_ROOT_RANK, cart_comm);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end gatherTiles");
  }
  /**********************************************************************************************************************/
  /*                      Implement each rank's local tile gather to master's rank global tile.                         */
  /*     The template T parameter is restricted to int or float type. You can choose the correct MPI datatype like:     */
  /*                                                                                                                    */
  /*  const MPI_Datatype localTileType  = std::is_same_v<T, int> ? localIntTileType    : localfloatTileType;            */
  /*  const MPI_Datatype globalTileType = std::is_same_v<T, int> ? globalFloatTileType : globalIntTileType;             */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::computeHaloZones(const float *oldTemp, float *newTemp)
{

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("computeHaloZones");
  }
  // vypočítat horní a spodní řádky

  // TODO: zjistit zda se jedná o kraj matice, aby se nepočítyl rohy

  // horní řádek
  if (neighbours[UP] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               2 * haloZoneSize, haloZoneSize,
               localCols - 2 * haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // spodní řádek
  if (neighbours[DOWN] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               2 * haloZoneSize, localRows,
               localCols - 2 * haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // levý sloupec
  if (neighbours[LEFT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               haloZoneSize, 2 * haloZoneSize,
               haloZoneSize, localRows - 2 * haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // pravý sloupec
  if (neighbours[RIGHT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               localCols, 2 * haloZoneSize,
               haloZoneSize, localRows - 2 * haloZoneSize,
               localCols + 2 * haloZoneSize);
  }

  // levý horní roh
  if (neighbours[UP] != MPI_PROC_NULL && neighbours[LEFT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               haloZoneSize, haloZoneSize,
               haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // pravý horní roh
  if (neighbours[UP] != MPI_PROC_NULL && neighbours[RIGHT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               localCols, haloZoneSize,
               haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // levý dolní roh
  if (neighbours[DOWN] != MPI_PROC_NULL && neighbours[LEFT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               haloZoneSize, localRows,
               haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  // pravý dolní roh
  if (neighbours[DOWN] != MPI_PROC_NULL && neighbours[RIGHT] != MPI_PROC_NULL)
  {
    updateTile(oldTemp, newTemp, localDomainParams.data(), localDomainMap.data(),
               localCols, localRows,
               haloZoneSize, haloZoneSize,
               localCols + 2 * haloZoneSize);
  }
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end computeHaloZones");
  }

  /**********************************************************************************************************************/
  /*  Compute new temperatures in halo zones, so that copy operations can be overlapped with inner region computation.  */
  /*                        Use updateTile method to compute new temperatures in halo zones.                            */
  /*                             TAKE CARE NOT TO COMPUTE THE SAME AREAS TWICE                                          */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::startHaloExchangeP2P(float *localData, std::array<MPI_Request, 8> &requests)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("startHaloExchangeP2P");
  }

  MPI_Isend(&localData[startHaloUpInside], 1, ROW_TYPE_FLOAT, neighbours[UP], 0, cart_comm, &requests[0]);
  MPI_Isend(&localData[startHaloLeftInside], 1, COL_TYPE_FLOAT, neighbours[LEFT], 0, cart_comm, &requests[1]);
  MPI_Isend(&localData[startHaloRightInside], 1, COL_TYPE_FLOAT, neighbours[RIGHT], 0, cart_comm, &requests[2]);
  MPI_Isend(&localData[startHaloDownInside], 1, ROW_TYPE_FLOAT, neighbours[DOWN], 0, cart_comm, &requests[3]);

  MPI_Irecv(&localData[startHaloUpOutside], 1, ROW_TYPE_FLOAT, neighbours[UP], 0, cart_comm, &requests[4]);
  MPI_Irecv(&localData[startHaloLeftOutside], 1, COL_TYPE_FLOAT, neighbours[LEFT], 0, cart_comm, &requests[5]);
  MPI_Irecv(&localData[startHaloRightOutside], 1, COL_TYPE_FLOAT, neighbours[RIGHT], 0, cart_comm, &requests[6]);
  MPI_Irecv(&localData[startHaloDownOutside], 1, ROW_TYPE_FLOAT, neighbours[DOWN], 0, cart_comm, &requests[7]);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end startHaloExchangeP2P");
  }
  /**********************************************************************************************************************/
  /*                       Start the non-blocking halo zones exchange using P2P communication.                          */
  /*                         Use the requests array to return the requests from the function.                           */
  /*                            Don't forget to set the empty requests to MPI_REQUEST_NULL.                             */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::startHaloExchangeRMA(float *localData, MPI_Win window)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("startHaloExchangeRMA");
  }
  MPI_Win_fence(0, window);

  MPI_Put(&localData[startHaloUpInside], 1, ROW_TYPE_FLOAT,
          neighbours[UP], startHaloDownOutside, 1, ROW_TYPE_FLOAT, window);

  MPI_Put(&localData[startHaloDownInside], 1, ROW_TYPE_FLOAT,
          neighbours[DOWN], startHaloUpOutside, 1, ROW_TYPE_FLOAT, window);

  MPI_Put(&localData[startHaloLeftInside], 1, COL_TYPE_FLOAT,
          neighbours[LEFT], startHaloRightOutside, 1, COL_TYPE_FLOAT, window);

  MPI_Put(&localData[startHaloRightInside], 1, COL_TYPE_FLOAT,
          neighbours[RIGHT], startHaloLeftOutside, 1, COL_TYPE_FLOAT, window);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end startHaloExchangeRMA");
  }
  /**********************************************************************************************************************/
  /*                       Start the non-blocking halo zones exchange using RMA communication.                          */
  /*                   Do not forget that you put/get the values to/from the target's opposite side                     */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::awaitHaloExchangeP2P(std::array<MPI_Request, 8> &requests)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("awaitHaloExchangeP2P");
  }

  MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end awaitHaloExchangeP2P");
  }
  /**********************************************************************************************************************/
  /*                       Wait for all halo zone exchanges to finalize using P2P communication.                        */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::awaitHaloExchangeRMA(MPI_Win window)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("awaitHaloExchangeRMA");
  }

  MPI_Win_fence(0, window);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end awaitHaloExchangeRMA");
  }
  /**********************************************************************************************************************/
  /*                       Wait for all halo zone exchanges to finalize using RMA communication.                        */
  /**********************************************************************************************************************/
}

void ParallelHeatSolver::run(std::vector<float, AlignedAllocator<float>> &outResult)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("run");
  }
  std::array<MPI_Request, 8> requestsP2P{};

  std::vector<int, AlignedAllocator<int>> globalDomainMap = mMaterialProps.getDomainMap();
  std::vector<float, AlignedAllocator<float>> globalDomainParams = mMaterialProps.getDomainParameters();
  std::vector<float, AlignedAllocator<float>> globalInitTemp = mMaterialProps.getInitialTemperature();

  /**********************************************************************************************************************/
  /*                                         Scatter initial data.                                                      */
  /**********************************************************************************************************************/

  if (DEBUG && mWorldRank == MPI_ROOT_RANK)
  {
    printf("===================== Print matrix of init TEMP after gather tiles =====================\n");
    printMatrix(globalInitTemp.data(), edgeSize, edgeSize);
    mpiFlush();
  }

  scatterTiles(globalInitTemp.data(), localTemp[OLD_TEMP]);
  scatterTiles(globalDomainMap.data(), localDomainMap.data());
  scatterTiles(globalDomainParams.data(), localDomainParams.data());

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printf("===================== Print matrix of old TEMP after scatter tiles =====================\n");
    printMatrix(localTemp[OLD_TEMP], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
    mpiFlush();
  }
  /**********************************************************************************************************************/
  /* Exchange halo zones of initial domain temperature and parameters using P2P communication. Wait for them to finish. */
  /**********************************************************************************************************************/

  startHaloExchangeP2P(localTemp[OLD_TEMP], requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);

  startHaloExchangeP2P(localDomainParams.data(), requestsP2P);
  awaitHaloExchangeP2P(requestsP2P);

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printf("===================== Print matrix of old TEMP after first exchange halo zone =====================\n");
    printMatrix(localTemp[OLD_TEMP], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
    mpiFlush();
  }

  /**********************************************************************************************************************/
  /*                            Copy initial temperature to the second buffer.                                          */
  /**********************************************************************************************************************/

  double startTime = MPI_Wtime();

  std::copy(localTemp[OLD_TEMP], localTemp[OLD_TEMP] + size_with_halo, localTemp[NEW_TEMP]);

  std::size_t oldIdx = OLD_TEMP;
  std::size_t newIdx = NEW_TEMP;

  // 3. Start main iterative simulation loop.
  for (std::size_t iter = 0; iter < mSimulationProps.getNumIterations(); ++iter)
  {
    oldIdx = iter % 2;       // Index of the buffer with old temperatures
    newIdx = (iter + 1) % 2; // Index of the buffer with new temperatures

    /**********************************************************************************************************************/
    /*                            Compute and exchange halo zones using P2P or RMA.                                       */
    /**********************************************************************************************************************/

    computeHaloZones(localTemp[oldIdx], localTemp[newIdx]);

    if (DEBUG && mWorldRank == PRINT_RANK)
    {
      printf("===================== Print matrix of new TEMP after compute halo zone (iter %lu) =====================\n", iter);
      printMatrix(localTemp[newIdx], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
      mpiFlush();
    }

    if (mSimulationProps.isRunParallelP2P())
    {
      startHaloExchangeP2P(localTemp[newIdx], requestsP2P);
    }
    else if (mSimulationProps.isRunParallelRMA())
    {
      startHaloExchangeRMA(localTemp[newIdx], HALO_WINDOW[newIdx]);
    }

    /**********************************************************************************************************************/
    /*                           Compute the rest of the tile. Use updateTile method.                                     */
    /**********************************************************************************************************************/
    if (DEBUG && mWorldRank == PRINT_RANK)
    {
      printEcho("compute center tile");
    }

    updateTile(localTemp[oldIdx], localTemp[newIdx],
               localDomainParams.data(), localDomainMap.data(),
               2 * haloZoneSize, 2 * haloZoneSize,                         // offset from the top and left
               localCols - 2 * haloZoneSize, localRows - 2 * haloZoneSize, // size of the tile
               localCols + 2 * haloZoneSize);                              // stride
    if (DEBUG && mWorldRank == PRINT_RANK)
    {
      printEcho("end compute center tile");
    }
    if (DEBUG && mWorldRank == PRINT_RANK)
    {
      printf("===================== Print matrix of new TEMP after compute center tile (iter %lu) =====================\n", iter);
      printMatrix(localTemp[newIdx], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
      mpiFlush();
    }
    /**********************************************************************************************************************/
    /*                            Wait for all halo zone exchanges to finalize.                                           */
    /**********************************************************************************************************************/
    if (mSimulationProps.isRunParallelP2P())
    {
      awaitHaloExchangeP2P(requestsP2P);
    }
    else if (mSimulationProps.isRunParallelRMA())
    {
      awaitHaloExchangeRMA(HALO_WINDOW[newIdx]);
    }

    if (shouldStoreData(iter))
    {
      /**********************************************************************************************************************/
      /*                          Store the data into the output file using parallel or sequential IO.                      */
      /**********************************************************************************************************************/

      // if (mSimulationProps.useParallelIO())
      //{
      //   storeDataIntoFileParallel(mFileHandle, iter, localTemp[NEW_TEMP].data());
      // }
      // else if (mpiGetCommRank(cart_comm) == MPI_ROOT_RANK)
      //{
      //   storeDataIntoFileSequential(mFileHandle, iter, localTemp[NEW_TEMP].data());
      // }
    }

    if (shouldPrintProgress(iter) && shouldComputeMiddleColumnAverageTemperature())
    {
      // printMatrix(localTemp[newIdx], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
      float middleVal = computeMiddleColumnAverageTemperatureParallel(localTemp[newIdx]);

      if (mpiGetCommRank(middle_col_comm) == MPI_ROOT_RANK)
      {
        printProgressReport(iter, middleVal);
      }
      /**********************************************************************************************************************/
      /*                 Compute and print middle column average temperature and print progress report.                     */
      /**********************************************************************************************************************/
    }

    if (DEBUG && mWorldRank == PRINT_RANK)
    {
      printf("===================== Print matrix of new TEMP on end one iter (iter %lu) =====================\n", iter);
      printMatrix(localTemp[newIdx], (localRows + 2 * haloZoneSize), (localCols + 2 * haloZoneSize));
      mpiFlush();
    }
  }

  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end run");
  }

  double elapsedTime = MPI_Wtime() - startTime;

  /**********************************************************************************************************************/
  /*                                     Gather final domain temperature.                                               */
  /**********************************************************************************************************************/

  gatherTiles(localTemp[newIdx], outResult.data());

  if (DEBUG && mWorldRank == MPI_ROOT_RANK)
  {
    printf("===================== Print matrix of final TEMP after gather tiles =====================\n");
    printMatrix(outResult.data(), edgeSize, edgeSize);
    mpiFlush();
  }

  /**********************************************************************************************************************/
  /*           Compute (sequentially) and report final middle column temperature average and print final report.        */
  /**********************************************************************************************************************/

  if (mpiGetCommRank(cart_comm) == MPI_ROOT_RANK)
  {
    float middleVal = computeMiddleColumnAverageTemperatureSequential(outResult.data());
    printFinalReport(elapsedTime, middleVal);
  }
}

bool ParallelHeatSolver::shouldComputeMiddleColumnAverageTemperature() const
{
  /**********************************************************************************************************************/
  /*                Return true if rank should compute middle column average temperature.                               */
  /**********************************************************************************************************************/

  return middle_col_comm != MPI_COMM_NULL;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureParallel(const float *localData) const
{
  /**********************************************************************************************************************/
  /*                  Implement parallel middle column average temperature computation.                                 */
  /*                      Use OpenMP directives to accelerate the local computations.                                   */
  /**********************************************************************************************************************/
  int middleIndex = (edgeSize / 2) % localCols;
  // printf("middleIndex: %d\n", middleIndex);

  float localmiddleColAvgTemp = 0.0f;

#pragma omp parallel for reduction(+ : localmiddleColAvgTemp)
  for (std::size_t i = haloZoneSize; i < localRows + haloZoneSize; ++i)
  {
    localmiddleColAvgTemp += localData[i * (localRows + 2 * haloZoneSize) + middleIndex + haloZoneSize];
    // printf("rank: %d, temp [%d]: %f\n", mpiGetCommRank(cart_comm), i * (localRows + 2 * haloZoneSize) + middleIndex + haloZoneSize, localData[i * (localRows + 2 * haloZoneSize) + middleIndex + haloZoneSize]);
  }

  float gloabalmiddleColAvgTemp = 0.0f;
  MPI_Reduce(&localmiddleColAvgTemp, &gloabalmiddleColAvgTemp, 1, MPI_FLOAT, MPI_SUM, MPI_ROOT_RANK, middle_col_comm);

  return gloabalmiddleColAvgTemp / edgeSize;
}

float ParallelHeatSolver::computeMiddleColumnAverageTemperatureSequential(const float *globalData) const
{
  /**********************************************************************************************************************/
  /*                  Implement sequential middle column average temperature computation.                               */
  /*                      Use OpenMP directives to accelerate the local computations.                                   */
  /**********************************************************************************************************************/
  float middleColAvgTemp{};

#pragma omp parallel for reduction(+ : middleColAvgTemp)
  for (std::size_t i = 0; i < edgeSize; ++i)
  {
    middleColAvgTemp += globalData[i * edgeSize + edgeSize / 2];
  }

  return middleColAvgTemp / edgeSize;
}

void ParallelHeatSolver::openOutputFileSequential()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("openOutputFileSequential");
  }
  if (mWorldRank == MPI_ROOT_RANK)
  {
    // Create the output file for sequential access.
    mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                            H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (!mFileHandle.valid())
    {
      throw std::ios::failure("Cannot create output file!");
    }

    hsize_t dims[2] = {(hsize_t)globalRows, (hsize_t)globalCols};
    hsize_t ldims[2] = {(edgeSize / globalRows), (edgeSize / globalCols)};

    hid_t filespace_id = H5Screate_simple(2, dims, NULL);
    hid_t memspace_id = H5Screate_simple(2, ldims, NULL);

    hid_t dataset_id = H5Dcreate2(mFileHandle, "DATASET", H5T_NATIVE_INT, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end openOutputFileSequential");
  }
}

void ParallelHeatSolver::storeDataIntoFileSequential(hid_t fileHandle,
                                                     std::size_t iteration,
                                                     const float *globalData)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("storeDataIntoFileSequential");
  }
  storeDataIntoFile(fileHandle, iteration, globalData);
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end storeDataIntoFileSequential");
  }
}

void ParallelHeatSolver::openOutputFileParallel()
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("openOutputFileParallel");
  }
#ifdef H5_HAVE_PARALLEL
  Hdf5PropertyListHandle faplHandle{};

  /**********************************************************************************************************************/
  /*                          Open output HDF5 file for parallel access with alignment.                                 */
  /*      Set up faplHandle to use MPI-IO and alignment. The handle will automatically release the resource.            */
  /**********************************************************************************************************************/

  mFileHandle = H5Fcreate(mSimulationProps.getOutputFileName(codeType).c_str(),
                          H5F_ACC_TRUNC,
                          H5P_DEFAULT,
                          faplHandle);
  if (!mFileHandle.valid())
  {
    throw std::ios::failure("Cannot create output file!");
  }

#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end openOutputFileParallel");
  }
}

void ParallelHeatSolver::storeDataIntoFileParallel(hid_t fileHandle,
                                                   [[maybe_unused]] std::size_t iteration,
                                                   [[maybe_unused]] const float *localData)
{
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("storeDataIntoFileParallel");
  }
  if (fileHandle == H5I_INVALID_HID)
  {
    return;
  }

#ifdef H5_HAVE_PARALLEL
  std::array gridSize{static_cast<hsize_t>(mMaterialProps.getEdgeSize()),
                      static_cast<hsize_t>(mMaterialProps.getEdgeSize())};

  // Create new HDF5 group in the output file
  std::string groupName = "Timestep_" + std::to_string(iteration / mSimulationProps.getWriteIntensity());

  Hdf5GroupHandle groupHandle(H5Gcreate(fileHandle, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

  {
    /**********************************************************************************************************************/
    /*                                Compute the tile offsets and sizes.                                                 */
    /*               Note that the X and Y coordinates are swapped (but data not altered).                                */
    /**********************************************************************************************************************/

    // Create new dataspace and dataset using it.
    static constexpr std::string_view dataSetName{"Temperature"};

    Hdf5PropertyListHandle datasetPropListHandle{};

    /**********************************************************************************************************************/
    /*                            Create dataset property list to set up chunking.                                        */
    /*                Set up chunking for collective write operation in datasetPropListHandle variable.                   */
    /**********************************************************************************************************************/

    Hdf5DataspaceHandle dataSpaceHandle(H5Screate_simple(2, gridSize.data(), nullptr));
    Hdf5DatasetHandle dataSetHandle(H5Dcreate(groupHandle, dataSetName.data(),
                                              H5T_NATIVE_FLOAT, dataSpaceHandle,
                                              H5P_DEFAULT, datasetPropListHandle,
                                              H5P_DEFAULT));

    Hdf5DataspaceHandle memSpaceHandle{};

    /**********************************************************************************************************************/
    /*                Create memory dataspace representing tile in the memory (set up memSpaceHandle).                    */
    /**********************************************************************************************************************/

    /**********************************************************************************************************************/
    /*              Select inner part of the tile in memory and matching part of the dataset in the file                  */
    /*                           (given by position of the tile in global domain).                                        */
    /**********************************************************************************************************************/

    Hdf5PropertyListHandle propListHandle{};

    /**********************************************************************************************************************/
    /*              Perform collective write operation, writting tiles from all processes at once.                        */
    /*                                   Set up the propListHandle variable.                                              */
    /**********************************************************************************************************************/

    H5Dwrite(dataSetHandle, H5T_NATIVE_FLOAT, memSpaceHandle, dataSpaceHandle, propListHandle, localData);
  }

  {
    // 3. Store attribute with current iteration number in the group.
    static constexpr std::string_view attributeName{"Time"};
    Hdf5DataspaceHandle dataSpaceHandle(H5Screate(H5S_SCALAR));
    Hdf5AttributeHandle attributeHandle(H5Acreate2(groupHandle, attributeName.data(),
                                                   H5T_IEEE_F64LE, dataSpaceHandle,
                                                   H5P_DEFAULT, H5P_DEFAULT));
    const double snapshotTime = static_cast<double>(iteration);
    H5Awrite(attributeHandle, H5T_IEEE_F64LE, &snapshotTime);
  }
#else
  throw std::runtime_error("Parallel HDF5 support is not available!");
#endif /* H5_HAVE_PARALLEL */
  if (DEBUG && mWorldRank == PRINT_RANK)
  {
    printEcho("end storeDataIntoFileParallel");
  }
}
