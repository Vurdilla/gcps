#ifndef VC3_PHYS_KSNP_SETUP
#define VC3_PHYS_KSNP_SETUP

#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>

#include "../include/general/stringext.cu"
#include "../include/types.cu"
#include "../include/general/INIReader.cu"


namespace vc3_phys{

/** KSNP target parameters 
**/
struct targetParameters
{
	flt2 targetDistanceMin; // minimum distance from the origin
	flt2 targetDistanceMax; // maximum distance from the origin
	flt2 targetAngleMin; // minimum anglular position, in degrees
	flt2 targetAngleMax; // maximum anglular position, in degrees
	flt2 targetRadiusMin; // minimum target radius
	flt2 targetRadiusMax; // maximum target radius
	flt2 targetWeightMin; // minimum target weight (number of first hits to capture)
	flt2 targetWeightMax; // maximum target weight (number of first hits to capture)
	int targetAppearType; // 0 - appear by time, 1 - appear after termination
	flt2 targetAppearTime; // loop appear time
	flt2 targetAppearDelay; // time delay before appear trigger and real appear of the target
	int targetTerminateType; // 0 - terminate by time, 1 - terminate after capture
	flt2 targetTerminateTime; // loop terminate time
	int targetResetType; // target reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center
	//   2 - reverse direction + one timestep backward
	flt2 betaTargetMultMin; // minimum beta multiplication coefficient when particle hits target
	flt2 betaTargetMultMax; // maximum beta multiplication coefficient when particle hits target
	bool targetResetSRtime; // is target hit resets stochastic reset time or not
	bool targetHomePotential; // does target hit switches on home potential sensing
};

/** KSNp setup data
**/
struct setupParameters
{
	// default constructor
	setupParameters();

	// Read parameters from an ini file
	__host__ int read(std::string filename);

	// Write parameters to an ini file
	__host__ int write(std::string filename, bool header = true);

	// searcher parameters
	int nParticles; // number of particles, default = 1
	int chemosensitivityModel; // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
	flt2 V0; // searcher velocity, default =  1
	bool keepV0Constant; // use equations with constant V0, default = false
	flt2 rotationalDiffusion; // rotationbal diffusion coefficient, default = 1.0
	flt2 Rscent; // scent raduis, default = 0.05
	flt2 beta0; // default = 0.005
	flt2 betaDecayTime; // default = 1.0
	flt2 chiRot; // default =  0.2
	flt2 chiTrans; // default =  0.015
	flt2 SC0; // scent noise level, default = 1.00

	// pairwise searcher interactions
	flt2 PPepsilon; // soft potential epsilon
	flt2 PPsigma; // soft potential sigma

	// system parameters
	flt2 boxSize; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	int boundaryType; // boundary: 0 - square, 1 - circle
	flt2 scentDecayTime; // scent decay time, default = 1.00
	int initialNTRtype; // initian next time time reset type: 0 - same as regular, 1 - random at [0.00, timedResetMeanTime)

	// home potential
	int homeType; // home type: 1 - parabolic potential, 2 - conical potential
	flt2 homeRadius; // radius of the circular area from which particles instantly resets to the center
	flt2 homePotentialKT; // strength of the home potential, translational strength
	flt2 homePotentialKR; // strength of the home potential, rotational strength

	// timed reset parameters
	int timedResetTimerType; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	int timedResetType; // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	flt2 timedResetMeanTime; // average time between resets, default = 50
	bool timedResetHomePotential; // does timed reset switches on home potential sensing

	// boundary parameters
	int boundaryResetType; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	flt2 betaBoundaryMult; // beta multiplication coefficient when particle hits boundary
	bool boundaryResetSRtime; // does boundary hit reset stochastic reset time or not
	bool boundaryHomePotential; // does boundary hit switches on home potential sensing

	// target parameters
	std::vector<targetParameters> targetList;

	// computation parameters
	int latticeSize; // number of nodes on each side of the grid, default = 200
	flt2 timeStep; // length of time step, default = 0.005
	flt2 Rscent_cutoffMultiplier; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	long long int scentDecayRescalingNsteps; // number of steps between inflation rescaling, defaul = 100000
	int gpuHistoryLength; // number of steps stored on GPU before it's flushed to CPU and analyzed

	// run parameters
	long long int nSteps; // total number of time steps in each iteration, default = 20000
	int nIterations; // total number of iterations, default = 100

	// output parameters
	flt2 cumulativeDataMinTime; // minimum time after which all cumulative data to be collected
	int cumulativePCblockSize; // block size for the cumulative particle concentration
	long long int PCReg_startstep;
	long long int PCReg_window;
	int PCReg_blockSize; // block size for the cumulative particle concentration
	long long int SCReg_startstep;
	long long int SCReg_window;
	int SCReg_blockSize; // block size for the cumulative particle concentration
	flt2 stepDrHistogram_min, stepDrHistogram_max; // limits of dtep dr histogram
	int stepDrHistogram_nbins; // number of bins in dtep dr histogram
	long long int totalScentReg_window; // regular time total scent will be averaged inside window in each iteration, in steps
	long long int totalScentReg_avgperiod; // regular time total scent will be averaged with this period in each iteration, in steps
	int PCrtheta_r_nbins; // number of bins in rtheta PC hist over radius
	flt2 PCrtheta_theta_binsize; // linear size of bins in rtheta PC hist over theta
	long long int PCrthetaReg_window; // regular time rtheta PC will be accumulated inside window in each iteration, in steps
	int PCtheta_nbins; // number of bins in theta PC hist
	flt2 PCtheta_rmin; // r threshold after which particles participates in theta PC hist
	flt2 PCtheta_rmax; // r threshold after which particles participates in theta PC hist
	long long int PCthetaReg_window; // regular time theta PC will be accumulated inside window in each iteration, in steps
	long long int HitCountReg_window; // regular time interval at which hit counts are stored, in steps
	int NTrj; // Number of first trajectories to output per iteration
	int EveryTrj; // Output every this trajectorie per iteration
	bool computeMSD; // Compute MSD of the trajectories
	int maxMSDLength; // length of MSD output range
	int MSDstep; // step count between the nearest MSD points
	flt2 PRH_min; // limits of path-rotation histogram
	flt2 PRH_max; // limits of path-rotation histogram
	int PRH_nbins; // number of bins in path-rotation histogram
	int PRH_step; // step between the nearest points for PRH
	int GTFPT_blockSize; // block size for GT FPT collection
	std::vector<flt2> GTFPT_Rtarget; // GT FPT target radius vector

}; // struct setupParameters

} //namespace vc3_phys


vc3_phys::setupParameters::setupParameters()
{
	// searcher parameters
	nParticles = 1; // number of particles, default = 1
	chemosensitivityModel = 0; // chemosensitivity model : 0 - ~gradient of SC, 1 - gradient of log(SC)
	V0 = 1; // searcher velocity, default =  1
	keepV0Constant = false; // use equations with constant V0, default = false
	rotationalDiffusion = 1.0; // rotationbal diffusion coefficient, default = 1.0
	Rscent = 0.05; // scent raduis, default = 0.05
	beta0 = 0.005; // default = 0.005
	betaDecayTime = 1.0; // default = 1.0
	chiRot = 0.2; // default =  0.2
	chiTrans = 0.015; // default =  0.015
	SC0 = 1.00; // scent noise level, default = 1.00

	// pairwise searcher interactions
	PPepsilon = 0.00; // soft potential epsilon
	PPsigma = 0.01; // soft potential sigma

	// system parameters
	boxSize = 1.00; // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	boundaryType = 0; // boundary: 0 - square, 1 - circle
	scentDecayTime = 1.00; // scent decay time, default = 1.00
	initialNTRtype = 0; // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)

	// home potential
	homeType = 0; // home type: 1 - parabolic potential, 2 - conical potential
	homeRadius = 0.1; // radius of the circular area at the origin called 'home'
	homePotentialKT = 0.0; // Strength of the home potential, translational strength
	homePotentialKR = 0.0; // Strength of the home potential, rotational strength

	// timed reset parameters
	timedResetTimerType = 0; // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	timedResetType = 0; // stochastic reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center,
	//   2 - reverse direction + one timestep backward
	timedResetMeanTime = 50; // average time between resets, default = 50
	timedResetHomePotential = false; // does timed reset switches on home potential sensing

	// boundary parameter	
	boundaryResetType = 0; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center
	//   2 - reverse direction + one timestep towards center
	betaBoundaryMult = 0.00; // beta multiplication coefficient when particle hits boundary
	boundaryResetSRtime = true; // is boundary hit resets stochastic reset time or not
	boundaryHomePotential = false; // does boundary hit switches on home potential sensing

	// target parameters

	// computation parameters
	latticeSize = 200; // number of nodes on each side of the grid, default = 200
	timeStep = 0.005; // length of time step, default = 0.005
	Rscent_cutoffMultiplier = 3.0; // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	scentDecayRescalingNsteps = 100000; // number of steps between inflation rescaling, default = 100000

	// run parameters
	nSteps = 10000; // total number of time steps in each iteration, default = 10000
	nIterations = 1; // total number of iterations, default = 1

	// output parameters
	cumulativeDataMinTime = 0.00;
	cumulativePCblockSize = 1;
	stepDrHistogram_min = 0.00; // limits of step dr histogram
	stepDrHistogram_max = 0.02; // limits of step dr histogram
	stepDrHistogram_nbins = 400; // number of bins in step dr histogram
	totalScentReg_window = 100000; // regular time total scent will be averaged inside window in each iteration, in steps
	totalScentReg_avgperiod = 1000000; // regular time total scent will be averaged with this period in each iteration, in steps
	PCrtheta_r_nbins = 10; // number of bins in rtheta particle concentration hist over radius
	PCrtheta_theta_binsize = 1.00; // linear size of bins in rtheta particle concentration hist over theta
	PCrthetaReg_window = 1000000; // regular time rtheta PC will be accumulated inside window in each iteration, in steps
	PCtheta_nbins = 10; // number of bins in theta PC hist
	PCtheta_rmin = 0.00; // r threshold after which particles participates in theta PC hist
	PCtheta_rmax = 0.00; // r threshold after which particles participates in theta PC hist
	PCthetaReg_window = 1000000; // regular time theta PC will be accumulated inside window in each iteration, in steps
	HitCountReg_window = 100000; // regular time interval at which hit counts are stored, in steps
	NTrj = 0;
	EveryTrj = 1;
	computeMSD = false;
	maxMSDLength = 1;
	MSDstep = 1;
	PRH_min = 0.00;
	PRH_max = 1.00;
	PRH_nbins = 1;
	PRH_step = 1;
	GTFPT_blockSize = 1;
	GTFPT_Rtarget.resize(0);
}

__host__ int vc3_phys::setupParameters::read(std::string filename)
{
	INIReader ini(filename);
	switch (ini.ParseError())
	{
	case 0: // parsed successfully
		break;
	default:
		std::cout << "Error parsing parameters file \"" << filename << "\"\n";
		return 1;
	}

	// searcher parameters
	nParticles = ini.GetInteger("Searcher", "nParticles", 1); // number of particles, default = 1
	chemosensitivityModel = ini.GetInteger("Searcher", "chemosensitivityModel", 1.0); // chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)
	V0 = ini.GetReal("Searcher", "V0", 1.0); // searcher velocity, default =  1
	keepV0Constant = ini.GetBoolean("Searcher", "keepV0Constant", false); // use equations with constant V0, default = false
	rotationalDiffusion = ini.GetReal("Searcher", "rotationalDiffusion", 1.0); // rotationbal diffusion coefficient, default = 1.0
	Rscent = ini.GetReal("Searcher", "Rscent", 0.05); // scent raduis, default = 0.05
	beta0 = ini.GetReal("Searcher", "beta0", 0.005); // default = 0.005
	betaDecayTime = ini.GetReal("Searcher", "betaDecayTime", 1.0); // default = 1.0
	chiRot = ini.GetReal("Searcher", "chiRot", 0.2); // default =  0.2
	chiTrans = ini.GetReal("Searcher", "chiTrans", 0.015); // default =  0.015
	SC0 = ini.GetReal("Searcher", "SC0", 1.0); // scent noise level

	// pairwise searcher interactions
	PPepsilon = ini.GetReal("Pairwise", "PPepsilon", 0.00); // soft potential epsilon
	PPsigma = ini.GetReal("Pairwise", "PPsigma", 0.01); // soft potential sigma

	// system parameters
	boxSize = ini.GetReal("System", "boxSize", 1.0); // the grid on which the concentration shall be calculated is 2L-by-2L, default = 1
	boundaryType = ini.GetInteger("System", "boundaryType", 0); // boundary: 0 - square, 1 - circle
	scentDecayTime = ini.GetReal("System", "scentDecayTime", 1.0); // scent decay time, default = 1.00
	initialNTRtype = ini.GetReal("System", "initialNTRtype", 0); // initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)

	// home potential
	homeType = ini.GetInteger("Home", "homeType", 0); // home type: 1 - parabolic potential, 2 - conical potential
	homeRadius = ini.GetReal("Home", "homeRadius", 0.1); // radius of the circular area at the origin called 'home'
	homePotentialKT = ini.GetReal("Home", "homePotentialKT", 0.0); // strength of the home potential, translational strength
	homePotentialKR = ini.GetReal("Home", "homePotentialKR", 0.0); // strength of the home potential, rotational strength

	// timed reset parameters
	timedResetTimerType = ini.GetInteger("TimedReset", "timedResetTimerType", 0); // timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution
	timedResetType = ini.GetInteger("TimedReset", "timedResetType", 0); // timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward
	timedResetMeanTime = ini.GetReal("TimedReset", "timedResetMeanTime", 50.0); // average time between timed resets, default =  50
	timedResetHomePotential = ini.GetBoolean("TimedReset", "timedResetHomePotential", false); // does timed reset switches on home potential sensing


	// boundary parameter	
	boundaryResetType = ini.GetInteger("Boundary", "boundaryResetType", 0);; // boundary reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep towards center
	betaBoundaryMult = ini.GetReal("Boundary", "betaBoundaryMult", 1.0); // beta multiplication coefficient when particle hits boundary
	boundaryResetSRtime = ini.GetBoolean("Boundary", "boundaryResetSRtime", false); // is boundary hit resets stochastic reset time or not
	boundaryHomePotential = ini.GetBoolean("Boundary", "boundaryHomePotential", false); // does boundary hit switches on home potential sensing

	// target list parameters
	int ntargets = ini.GetInteger("TargetList", "targetNtargets", 0);
	targetList.resize(ntargets);
	for (int nt = 0; nt < ntargets; nt++)
	{
		std::string tname = "Target" + vc3_general::itoa(nt + 1);
		targetList[nt].targetDistanceMin = ini.GetReal(tname, "targetDistanceMin", 0.00);
		targetList[nt].targetDistanceMax = ini.GetReal(tname, "targetDistanceMax", 0.00);
		targetList[nt].targetAngleMin = ini.GetReal(tname, "targetAngleMin", 0.00);
		targetList[nt].targetAngleMax = ini.GetReal(tname, "targetAngleMax", 0.00);
		targetList[nt].targetRadiusMin = ini.GetReal(tname, "targetRadiusMin", 0.00);
		targetList[nt].targetRadiusMax = ini.GetReal(tname, "targetRadiusMax", 0.00);
		targetList[nt].targetWeightMin = ini.GetReal(tname, "targetWeightMin", 0.00);
		targetList[nt].targetWeightMax = ini.GetReal(tname, "targetWeightMax", 0.00);
		targetList[nt].targetAppearType = ini.GetInteger(tname, "targetAppearType", 0);
		targetList[nt].targetAppearTime = ini.GetReal(tname, "targetAppearTime", 0);
		targetList[nt].targetAppearDelay = ini.GetReal(tname, "targetAppearDelay", 0.00);
		targetList[nt].targetTerminateType = ini.GetInteger(tname, "targetTerminateType", 0.00);
		targetList[nt].targetTerminateTime = ini.GetReal(tname, "targetTerminateTime", 0.00);
		targetList[nt].targetResetType = ini.GetInteger(tname, "targetResetType", 0.00);
		targetList[nt].betaTargetMultMin = ini.GetReal(tname, "betaTargetMultMin", 0.00);
		targetList[nt].betaTargetMultMax = ini.GetReal(tname, "betaTargetMultMax", 0.00);
		targetList[nt].targetResetSRtime = ini.GetBoolean(tname, "targetResetSRtime", true);
		targetList[nt].targetHomePotential = ini.GetBoolean(tname, "targetHomePotential", false);
	}

	// computations parameters
	latticeSize = ini.GetInteger("Computations", "latticeSize", 200); // number of nodes on each side of the grid, default = 200
	timeStep = ini.GetReal("Computations", "timeStep", 0.005); // length of time step, default = 0.005
	Rscent_cutoffMultiplier = ini.GetReal("Computations", "Rscent_cutoffMultiplier", 3.0); // scent radius cutoff multiplier (Rscent_cutoff = Rscent * Rscent_cutoffMultiplier), default = 3.0
	scentDecayRescalingNsteps = ini.GetLLInteger("Computations", "scentDecayRescalingNsteps", 100000); // number of steps between inflation rescaling, default = 100000
	gpuHistoryLength = ini.GetInteger("Computations", "gpuHistoryLength", 1000); // number of steps stored on GPU before it's flushed to CPU and analyzed

	// run parameters
	nSteps = ini.GetLLInteger("Run", "nSteps", 10000); // total number of time steps in each iteration, default = 10000
	nIterations = ini.GetInteger("Run", "nIterations", 1); // total number of iterations, default = 1

	// output parameters
	cumulativeDataMinTime = ini.GetReal("Output", "cumulativeDataMinTime", 0);
	cumulativePCblockSize = ini.GetInteger("Output", "cumulativePCblockSize", 1);
	PCReg_startstep = ini.GetLLInteger("Output", "PCReg_startstep", 0);
	PCReg_window = ini.GetLLInteger("Output", "PCReg_window", 0);
	PCReg_blockSize = ini.GetInteger("Output", "PCReg_blockSize", 1);
	SCReg_startstep = ini.GetLLInteger("Output", "SCReg_startstep", 0);
	SCReg_window = ini.GetLLInteger("Output", "SCReg_window", 0);
	SCReg_blockSize = ini.GetInteger("Output", "SCReg_blockSize", 1);
	stepDrHistogram_min = ini.GetReal("Output", "stepDrHistogram_min", 0);
	stepDrHistogram_max = ini.GetReal("Output", "stepDrHistogram_max", 0);
	stepDrHistogram_nbins = ini.GetInteger("Output", "stepDrHistogram_nbins", 0);
	totalScentReg_window = ini.GetLLInteger("Output", "totalScentReg_window", 0);
	totalScentReg_avgperiod = ini.GetLLInteger("Output", "totalScentReg_avgperiod", 0);
	PCrtheta_r_nbins = ini.GetInteger("Output", "PCrtheta_r_nbins", 10);
	PCrtheta_theta_binsize = ini.GetReal("Output", "PCrtheta_theta_binsize", 1.00);
	PCrthetaReg_window = ini.GetLLInteger("Output", "PCrthetaReg_window", 0);
	PCtheta_nbins = ini.GetInteger("Output", "PCtheta_nbins", 0);
	PCtheta_rmin = ini.GetReal("Output", "PCtheta_rmin", 0);
	PCtheta_rmax = ini.GetReal("Output", "PCtheta_rmax", 0);
	PCthetaReg_window = ini.GetLLInteger("Output", "PCthetaReg_window", 0);
	HitCountReg_window = ini.GetLLInteger("Output", "HitCountReg_window", 0);
	NTrj = ini.GetInteger("Output", "NTrj", 0);
	EveryTrj = ini.GetInteger("Output", "EveryTrj", 1);
	computeMSD = ini.GetBoolean("Output", "computeMSD", false);
	maxMSDLength = ini.GetInteger("Output", "maxMSDLength", 1);
	MSDstep = ini.GetInteger("Output", "MSDstep", 1);
	PRH_min = ini.GetReal("Output", "PRH_min", 0.0);
	PRH_max = ini.GetReal("Output", "PRH_max", 1.0);
	PRH_nbins = ini.GetInteger("Output", "PRH_nbins", 1);
	PRH_step = ini.GetInteger("Output", "PRH_step", 1);
	GTFPT_blockSize = ini.GetInteger("Output", "GTFPT_blockSize", 1);
	std::string s = ini.Get("Output", "GTFPT_Rtarget", "");
	vc3_general::stodv(s, &GTFPT_Rtarget, ',');

	std::cout << "\nParameter file \"" << filename << "\" read successfully\n";
	return 0;
}

__host__ int vc3_phys::setupParameters::write(std::string filename, bool header)
{
	std::ofstream f(filename.c_str());
	if (!f)
	{
		std::cout << "\nCan not write parameter file \"" << filename << "\"\n";
		return 1;
	}

	if (header) f << "; KSNP-cuda-v7.1 input parameters\n";
	f << "\n";
	f << "[Searcher]\n";
	f << "nParticles = " << nParticles << "\t; number of particles, default = 1\n";
	f << "chemosensitivityModel = " << chemosensitivityModel << "\t; chemosensitivity model: 0 - ~ gradient of SC, 1 - gradient of log(SC)\n";
	f << "V0 = " << V0 << "\t; searcher velocity\n";
	f << "keepV0Constant = " << keepV0Constant << "\t; use equations with constant V0\n";
	f << "rotationalDiffusion = " << rotationalDiffusion << "\t; rotationbal diffusion coefficient\n";
	f << "Rscent = " << Rscent << "\t; scent raduis\n";
	f << "beta0 = " << beta0 << "\t; default scent secretion rate\n";
	f << "betaDecayTime = " << betaDecayTime << "\t; scent secretion rate relaxation time\n";
	f << "chiRot = " << chiRot << "\t; \n";
	f << "chiTrans = " << chiTrans << "\t; \n";
	f << "SC0 = " << SC0 << "\t; scent noise level\n";
	f << "\n";
	f << "[Pairwise]\n";
	f << "PPepsilon = " << PPepsilon << "\t; soft potential epsilon\n";
	f << "PPsigma = " << PPsigma << "\t; soft potential sigma\n";
	f << "\n";
	f << "[System]\n";
	f << "boxSize = " << boxSize << "\t; the grid on which the concentration shall be calculated is 2L-by-2L\n";
	f << "boundaryType = " << boundaryType << "\t; boundary: 0 - square, 1 - circle\n";
	f << "scentDecayTime = " << scentDecayTime << "\t; scent decay time\n";
	f << "initialNTRtype = " << initialNTRtype << "\t; initian next time time reset type: 0 - same as regular, 1 - random at (0, timedResetMeanTime)\n";
	f << "\n";
	f << "[Home]\n";
	f << "homeType = " << homeType << "\t; home type: 1 - parabolic potential, 2 - conical potential\n";
	f << "homeRadius = " << homeRadius << "\t; radius of the circular area at the origin called 'home'\n";
	f << "homePotentialKT = " << homePotentialKT << "\t; strength of the home potential, translational strength\n";
	f << "homePotentialKR = " << homePotentialKR << "\t; strength of the home potential, rotational strength\n";
	f << "\n";
	f << "[TimedReset]\n";
	f << "timedResetTimerType = " << timedResetTimerType << "\t; timed reset timer type: 0 - regular time intervals (=meanResettingTime), 1 - stochastic time intervals with exp distribution\n";
	f << "timedResetType = " << timedResetType << "\t; timed reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
	f << "timedResetMeanTime = " << timedResetMeanTime << "\t; average time between timed resets\n";
	f << "timedResetHomePotential = " << timedResetHomePotential << "\t; does timed reset switches on home potential sensing\n";
	f << "\n";
	f << "[Boundary]\n";
	f << "boundaryResetType = " << boundaryResetType << "\t; boundary reset type: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
	f << "betaBoundaryMult = " << betaBoundaryMult << "\t; beta multiplication coefficient when particle hits boundary\n";
	f << "boundaryResetSRtime = " << boundaryResetSRtime << "\t; is boundary hit resets stochastic reset time or not\n";
	f << "boundaryHomePotential = " << boundaryHomePotential << "\t; does boundary hit switches on home potential sensing\n";
	f << "\n";
	f << "[TargetList]\n";
	f << "targetNtargets = " << targetList.size() << "\t; number of targets in the list\n";
	f << "\n";
	for (int nt = 0; nt < targetList.size(); nt++)
	{
		std::string tname = "Target" + vc3_general::itoa(nt + 1);
		f << "[" << tname << "]\n";
		f << "targetDistanceMin = " << targetList[nt].targetDistanceMin << "\t; minimum distance from the origin\n";
		f << "targetDistanceMax = " << targetList[nt].targetDistanceMax << "\t; maximum distance from the origin\n";
		f << "targetAngleMin = " << targetList[nt].targetAngleMin << "\t; minimum anglular position, in degrees\n";
		f << "targetAngleMax = " << targetList[nt].targetAngleMax << "\t; maximum anglular position, in degrees\n";
		f << "targetRadiusMin = " << targetList[nt].targetRadiusMin << "\t; minimum target radius\n";
		f << "targetRadiusMax = " << targetList[nt].targetRadiusMax << "\t; maximum target radius\n";
		f << "targetWeightMin = " << targetList[nt].targetWeightMin << "\t; minimum target weight (number of first hits to capture)\n";
		f << "targetWeightMax = " << targetList[nt].targetWeightMax << "\t; maximum target weight (number of first hits to capture)\n";
		f << "targetAppearType = " << targetList[nt].targetAppearType << "\t; 0 - appear by time, 1 - appear after termination\n";
		f << "targetAppearTime = " << targetList[nt].targetAppearTime << "\t; loop appear time\n";
		f << "targetAppearDelay = " << targetList[nt].targetAppearDelay << "\t; time delay before appear trigger and real appear of the target\n";
		f << "targetTerminateType = " << targetList[nt].targetTerminateType << "\t; 0 - terminate by time, 1 - terminate after capture\n";
		f << "targetTerminateTime = " << targetList[nt].targetTerminateTime << "\t; loop terminate time\n";
		f << "targetResetType = " << targetList[nt].targetResetType << "\t; target reset: 0 - position to the center + random direction, 1 - direction to the center + one timestep towards center, 2 - reverse direction + one timestep backward\n";
		f << "betaTargetMultMin = " << targetList[nt].betaTargetMultMin << "\t; minimum beta multiplication coefficient when particle hits target\n";
		f << "betaTargetMultMax = " << targetList[nt].betaTargetMultMax << "\t; maximum beta multiplication coefficient when particle hits target\n";
		f << "targetResetSRtime = " << targetList[nt].targetResetSRtime << "\t; is target hit resets stochastic reset time or not\n";
		f << "targetHomePotential = " << targetList[nt].targetHomePotential << "\t; does target hit switches on home potential sensing\n";
		f << "\n";
	}
	f << "[Computations]\n";
	f << "latticeSize = " << latticeSize << "\t; number of nodes on each side of the grid\n";
	f << "timeStep = " << timeStep << "\t; length of time step\n";
	f << "Rscent_cutoffMultiplier = " << Rscent_cutoffMultiplier << "\t; scent radius cutoff multiplier\n";
	f << "scentDecayRescalingNsteps = " << scentDecayRescalingNsteps << "\t; number of steps between inflation rescaling\n";
	f << "gpuHistoryLength = " << gpuHistoryLength << "\t; number of steps stored on GPU before it's flushed to CPU and analyzed\n";
	f << "\n";
	f << "[Run]\n";
	f << "nSteps = " << nSteps << "\t; total number of time steps in each iteration\n";
	f << "nIterations = " << nIterations << "\t; total number of iterations\n";
	f << "\n";
	f << "[Output]\n";
	f << "cumulativeDataMinTime = " << cumulativeDataMinTime << "\t; minimum time after which all cumulative data to be collected\n";
	f << "cumulativePCblockSize = " << cumulativePCblockSize << "\t; block size for the cumulative particle concentration\n";
	f << "PCReg_startstep = " << PCReg_startstep << "\t; \n";
	f << "PCReg_window = " << PCReg_window << "\t; \n";
	f << "PCReg_blockSize = " << PCReg_blockSize << "\t; \n";
	f << "SCReg_startstep = " << SCReg_startstep << "\t; \n";
	f << "SCReg_window = " << SCReg_window << "\t; \n";
	f << "SCReg_blockSize = " << SCReg_blockSize << "\t; \n";
	f << "stepDrHistogram_min = " << stepDrHistogram_min << "\t; limits of step dr histogram\n";
	f << "stepDrHistogram_max = " << stepDrHistogram_max << "\t; limits of step dr histogram\n";
	f << "stepDrHistogram_nbins = " << stepDrHistogram_nbins << "\t; number of bins in dtep dr histogram\n";
	f << "totalScentReg_window = " << totalScentReg_window << "\t; regular time total scent will be averaged inside window in each iteration, in steps\n";
	f << "totalScentReg_avgperiod = " << totalScentReg_avgperiod << "\t; regular time total scent will be averaged with this period in each iteration, in steps\n";
	f << "PCrtheta_r_nbins = " << PCrtheta_r_nbins << "\t; number of bins in rtheta PC hist over radius\n";
	f << "PCrtheta_theta_binsize = " << PCrtheta_theta_binsize << "\t; linear size of bins in rtheta PC hist over theta\n";
	f << "PCrthetaReg_window = " << PCrthetaReg_window << "\t; regular time rtheta PC will be accumulated inside window in each iteration, in steps\n";
	f << "PCtheta_nbins = " << PCtheta_nbins << "\t; number of bins in theta PC hist\n";
	f << "PCtheta_rmin = " << PCtheta_rmin << "\t; r threshold after which particles participates in theta PC hist\n";
	f << "PCtheta_rmax = " << PCtheta_rmax << "\t; r threshold after which particles participates in theta PC hist\n";
	f << "PCthetaReg_window = " << PCthetaReg_window << "\t; regular time theta PC will be accumulated inside window in each iteration, in steps\n";
	f << "HitCountReg_window = " << HitCountReg_window << "\t; regular time interval at which hit counts are stored, in steps\n";
	f << "NTrj = " << NTrj << "\t; number of first trajectories to output per iteration\n";
	f << "EveryTrj = " << EveryTrj << "\t; output every this trajectorie per iteration\n";
	f << "computeMSD = " << computeMSD << "\t; Compute MSD of the trajectories\n";
	f << "maxMSDLength = " << maxMSDLength << "\t; length of MSD output range\n";
	f << "MSDstep = " << MSDstep << "\t; step count between the nearest MSD points\n";
	f << "PRH_min = " << PRH_min << "\t; limits of path-rotation histogram\n";
	f << "PRH_max = " << PRH_max << "\t; limits of path-rotation histogram\n";
	f << "PRH_nbins = " << PRH_nbins << "\t; number of bins in path-rotation histogram\n";
	f << "PRH_step = " << PRH_step << "\t; step between the nearest points for PRH\n";
	f << "GTFPT_blockSize = " << GTFPT_blockSize << "\t; block size for GT FPT collection\n";
	f << "GTFPT_Rtarget = ";
	for (int q = 0; q < GTFPT_Rtarget.size(); q++)
	{
		if (q > 0) f << ", ";
		f << GTFPT_Rtarget[q];
	}
	f << "\t; GT FPT target radius vector\n";
	f << "\n";

	f.close();
	std::cout << "\nParameter file \"" << filename << "\" written successfully\n";
	return 0;
}


#endif
