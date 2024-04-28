/**
 * @file    ParallelHeatSolver.hpp
 *
 * @author  Name Surname <xlogin00@fit.vutbr.cz>
 *
 * @brief   Course: PPP 2023/2024 - Project 1
 *          This file contains implementation of parallel heat equation solver
 *          using MPI/OpenMP hybrid approach.
 *
 * @date    2024-02-23
 */

#ifndef PARALLEL_HEAT_SOLVER_HPP
#define PARALLEL_HEAT_SOLVER_HPP

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <mpi.h>

#include "AlignedAllocator.hpp"
#include "Hdf5Handle.hpp"
#include "HeatSolverBase.hpp"

#define MPI_ROOT_RANK 0

#define UP 0
#define DOWN 1
#define LEFT 2
#define RIGHT 3

#define OLD_TEMP 0
#define NEW_TEMP 1

/**
 * @brief The ParallelHeatSolver class implements parallel MPI based heat
 *        equation solver in 2D using 2D block grid decomposition.
 */
class ParallelHeatSolver : public HeatSolverBase
{
public:
  ParallelHeatSolver(const SimulationProperties &simulationProps, const MaterialProperties &materialProps);

  using HeatSolverBase::HeatSolverBase;

  virtual ~ParallelHeatSolver() override;

  using HeatSolverBase::operator=;

  virtual void run(std::vector<float, AlignedAllocator<float>> &outResult) override;

protected:
  std::array<float *, 2> localTemp;
  std::vector<float, AlignedAllocator<float>> localDomainParams;
  std::vector<int, AlignedAllocator<int>> localDomainMap;

  int globalCols;
  int globalRows;

  float airflowRate;
  float coolerTemp;

  int size_with_halo;

  int startHaloUpInside;
  int startHaloDownInside;
  int startHaloLeftInside;
  int startHaloRightInside;

  int startHaloUpOutside;
  int startHaloDownOutside;
  int startHaloLeftOutside;
  int startHaloRightOutside;

  size_t localCols;
  size_t localRows;

  size_t edgeSize;

  int neighbours[4];

  MPI_Comm cart_comm;
  MPI_Comm middle_col_comm;

  MPI_Datatype TILE_TYPE_INT;
  MPI_Datatype TILE_TYPE_HALO_INT;
  MPI_Datatype BLOCK_TYPE_INT;

  MPI_Datatype TILE_TYPE_FLOAT;
  MPI_Datatype TILE_TYPE_HALO_FLOAT;
  MPI_Datatype BLOCK_TYPE_FLOAT;

  MPI_Datatype ROW_TYPE_INT;
  MPI_Datatype ROW_TYPE_FLOAT;

  MPI_Datatype COL_TYPE_INT;
  MPI_Datatype COL_TYPE_FLOAT;

  float *distMatrix{};
  MPI_Win HALO_WINDOW[2];

private:
  /**
   * @brief Get type of the code.
   * @return Returns type of the code.
   */
  std::string_view
  getCodeType() const override;

  /**
   * @brief Initialize the grid topology.
   */
  void initGridTopology();

  /**
   * @brief Deinitialize the grid topology.
   */
  void deinitGridTopology();

  /**
   * @brief Initialize variables and MPI datatypes for data scattering and gathering.
   */
  void initDataDistribution();

  /**
   * @brief Deinitialize variables and MPI datatypes for data scattering and gathering.
   */
  void deinitDataDistribution();

  /**
   * @brief Allocate memory for local tiles.
   */
  void allocLocalTiles();

  /**
   * @brief Deallocate memory for local tiles.
   */
  void deallocLocalTiles();

  /**
   * @brief Initialize variables and MPI datatypes for halo exchange.
   */
  void initHaloExchange();

  /**
   * @brief Deinitialize variables and MPI datatypes for halo exchange.
   */
  void deinitHaloExchange();

  /**
   * @brief Scatter global data to local tiles.
   * @tparam T Type of the data to be scattered. Must be either float or int.
   * @param globalData Global data to be scattered.
   * @param localData  Local data to be filled with scattered values.
   */
  template <typename T>
  void scatterTiles(const T *globalData, T *localData);

  /**
   * @brief Gather local tiles to global data.
   * @tparam T Type of the data to be gathered. Must be either float or int.
   * @param localData  Local data to be gathered.
   * @param globalData Global data to be filled with gathered values.
   */
  template <typename T>
  void gatherTiles(const T *localData, T *globalData);

  /**
   * @brief Compute temperature of the next iteration in the halo zones.
   * @param oldTemp Old temperature values.
   * @param newTemp New temperature values.
   */
  void computeHaloZones(const float *oldTemp, float *newTemp);

  /**
   * @brief Start halo exchange using point-to-point communication.
   * @param localData Local data to be exchanged.
   * @param request   Array of MPI_Request objects to be filled with requests.
   */
  void startHaloExchangeP2P(float *localData, std::array<MPI_Request, 8> &request);

  /**
   * @brief Await halo exchange using point-to-point communication.
   * @param request Array of MPI_Request objects to be awaited.
   */
  void awaitHaloExchangeP2P(std::array<MPI_Request, 8> &request);

  /**
   * @brief Start halo exchange using RMA communication.
   * @param localData Local data to be exchanged.
   * @param window    MPI_Win object to be used for RMA communication.
   */
  void startHaloExchangeRMA(float *localData, MPI_Win window);

  /**
   * @brief Await halo exchange using RMA communication.
   * @param window MPI_Win object to be used for RMA communication.
   */
  void awaitHaloExchangeRMA(MPI_Win window);

  /**
   * @brief Computes global average temperature of middle column across
   *        processes in "mGridMiddleColComm" communicator.
   *        NOTE: All ranks in the communicator *HAVE* to call this method.
   * @param localData Data of the local tile.
   * @return Returns average temperature over middle of all tiles in the communicator.
   */
  float computeMiddleColumnAverageTemperatureParallel(const float *localData) const;

  /**
   * @brief Computes global average temperature of middle column of the domain
   *        using values collected to MASTER rank.
   *        NOTE: Only single RANK needs to call this method.
   * @param globalData Simulation state collected to the MASTER rank.
   * @return Returns the average temperature.
   */
  float computeMiddleColumnAverageTemperatureSequential(const float *globalData) const;

  /**
   * @brief Opens output HDF5 file for sequential access by MASTER rank only.
   *        NOTE: Only MASTER (rank = 0) should call this method.
   */
  void openOutputFileSequential();

  /**
   * @brief Stores current state of the simulation into the output file.
   *        NOTE: Only MASTER (rank = 0) should call this method.
   * @param fileHandle HDF5 file handle to be used for the writting operation.
   * @param iteration  Integer denoting current iteration number.
   * @param data       Square 2D array of edgeSize x edgeSize elements containing
   *                   simulation state to be stored in the file.
   */
  void storeDataIntoFileSequential(hid_t fileHandle, std::size_t iteration, const float *globalData);

  /**
   * @brief Opens output HDF5 file for parallel/cooperative access.
   *        NOTE: This method *HAS* to be called from all processes in the communicator.
   */
  void openOutputFileParallel();

  /**
   * @brief Stores current state of the simulation into the output file.
   *        NOTE: All processes which opened the file HAVE to call this method collectively.
   * @param fileHandle HDF5 file handle to be used for the writting operation.
   * @param iteration  Integer denoting current iteration number.
   * @param localData  Local 2D array (tile) of mLocalTileSize[0] x mLocalTileSize[1] elements
   *                   to be stored at tile specific position in the output file.
   *                   This method skips halo zones of the tile and stores only relevant data.
   */
  void storeDataIntoFileParallel(hid_t fileHandle, std::size_t iteration, const float *localData);

  /**
   * @brief Determines if the process should compute average temperature of the middle column.
   * @return Returns true if the process should compute average temperature of the middle column.
   */
  bool shouldComputeMiddleColumnAverageTemperature() const;

  /// @brief Code type string.
  static constexpr std::string_view codeType{"par"};

  /// @brief Size of the halo zone.
  static constexpr std::size_t haloZoneSize{2};

  /// @brief Process rank in the global communicator (MPI_COMM_WORLD).
  int mWorldRank{};

  /// @brief Total number of processes in MPI_COMM_WORLD.
  int mWorldSize{};

  /// @brief Output file handle (parallel or sequential).
  Hdf5FileHandle mFileHandle{};
};

#endif /* PARALLEL_HEAT_SOLVER_HPP */
