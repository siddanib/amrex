#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H> //For the method most common at time of writing
#include <AMReX_PlotFileUtil.H> //For ploting the MultiFab
#include <AMReX_Particles.H>
#include <mpi.h>
#include <AMReX_MPMD.H>

using namespace amrex;

static constexpr int NSR = 4;
static constexpr int NSI = 3;
static constexpr int NAR = 2;
static constexpr int NAI = 1;

int num_runtime_real = 0;
int num_runtime_int = 0;

void get_position_unit_cell(Real* r, const IntVect& nppc, int i_part)
{
        int nx = nppc[0];
#if AMREX_SPACEDIM >= 2
        int ny = nppc[1];
#else
        int ny = 1;
#endif
#if AMREX_SPACEDIM == 3
        int nz = nppc[2];
#else
        int nz = 1;
#endif

        AMREX_D_TERM(int ix_part = i_part/(ny * nz);,
                     int iy_part = (i_part % (ny * nz)) % ny;,
                     int iz_part = (i_part % (ny * nz)) / ny;)

        AMREX_D_TERM(r[0] = (0.5+ix_part)/nx;,
                     r[1] = (0.5+iy_part)/ny;,
                     r[2] = (0.5+iz_part)/nz;)
}

class TestParticleContainer
    : public amrex::ParticleContainer<NSR, NSI, NAR, NAI>
{

public:

    TestParticleContainer (const amrex::Geometry& a_geom,
                           const amrex::DistributionMapping& a_dmap,
                           const amrex::BoxArray& a_ba)
        : amrex::ParticleContainer<NSR, NSI, NAR, NAI>(a_geom, a_dmap, a_ba)
    {
        for (int i = 0; i < num_runtime_real; ++i)
        {
            AddRealComp(true);
        }
        for (int i = 0; i < num_runtime_int; ++i)
        {
            AddIntComp(true);
        }
    }

    void RedistributeLocal ()
    {
        const int lev_min = 0;
        const int lev_max = 0;
        const int nGrow = 0;
        const int local = 1;
        Redistribute(lev_min, lev_max, nGrow, local);
    }

    void InitParticles (const amrex::IntVect& a_num_particles_per_cell)
    {
        BL_PROFILE("InitParticles");

        const int lev = 0;  // only add particles on level 0
        const Real* dx = Geom(lev).CellSize();
        const Real* plo = Geom(lev).ProbLo();

        const int num_ppc = AMREX_D_TERM( a_num_particles_per_cell[0],
                                         *a_num_particles_per_cell[1],
                                         *a_num_particles_per_cell[2]);

        for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
        {
            const Box& tile_box  = mfi.tilebox();

            Gpu::HostVector<ParticleType> host_particles;
            std::array<Gpu::HostVector<ParticleReal>, NAR> host_real;
            std::array<Gpu::HostVector<int>, NAI> host_int;

            std::vector<Gpu::HostVector<ParticleReal> > host_runtime_real(NumRuntimeRealComps());
            std::vector<Gpu::HostVector<int> > host_runtime_int(NumRuntimeIntComps());

            for (IntVect iv = tile_box.smallEnd(); iv <= tile_box.bigEnd(); tile_box.next(iv))
            {
                for (int i_part=0; i_part<num_ppc;i_part++) {
                    Real r[AMREX_SPACEDIM];
                    get_position_unit_cell(r, a_num_particles_per_cell, i_part);

                    ParticleType p;
                    p.id()  = ParticleType::NextID();
                    p.cpu() = ParallelDescriptor::MyProc();
                    p.pos(0) = static_cast<ParticleReal> (plo[0] + (iv[0] + r[0])*dx[0]);
#if AMREX_SPACEDIM > 1
                    p.pos(1) = static_cast<ParticleReal> (plo[1] + (iv[1] + r[1])*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                    p.pos(2) = static_cast<ParticleReal> (plo[2] + (iv[2] + r[2])*dx[2]);
#endif

                    for (int i = 0; i < NSR; ++i) { p.rdata(i) = ParticleReal(p.id()); }
                    for (int i = 0; i < NSI; ++i) { p.idata(i) = int(p.id()); }

                    host_particles.push_back(p);
                    for (int i = 0; i < NAR; ++i) {
                        host_real[i].push_back(ParticleReal(p.id()));
                    }
                    for (int i = 0; i < NAI; ++i) {
                        host_int[i].push_back(int(p.id()));
                    }
                    for (int i = 0; i < NumRuntimeRealComps(); ++i) {
                        host_runtime_real[i].push_back(ParticleReal(p.id()));
                    }
                    for (int i = 0; i < NumRuntimeIntComps(); ++i) {
                        host_runtime_int[i].push_back(int(p.id()));
                    }
                }
            }

            auto& particle_tile = DefineAndReturnParticleTile(lev, mfi.index(), mfi.LocalTileIndex());
            auto old_size = particle_tile.GetArrayOfStructs().size();
            auto new_size = old_size + host_particles.size();
            particle_tile.resize(new_size);

            Gpu::copyAsync(Gpu::hostToDevice,
                           host_particles.begin(),
                           host_particles.end(),
                           particle_tile.GetArrayOfStructs().begin() + old_size);

            auto& soa = particle_tile.GetStructOfArrays();
            for (int i = 0; i < NAR; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_real[i].begin(),
                               host_real[i].end(),
                               soa.GetRealData(i).begin() + old_size);
            }

            for (int i = 0; i < NAI; ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_int[i].begin(),
                               host_int[i].end(),
                               soa.GetIntData(i).begin() + old_size);
            }
            for (int i = 0; i < NumRuntimeRealComps(); ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_real[i].begin(),
                               host_runtime_real[i].end(),
                               soa.GetRealData(NAR+i).begin() + old_size);
            }

            for (int i = 0; i < NumRuntimeIntComps(); ++i)
            {
                Gpu::copyAsync(Gpu::hostToDevice,
                               host_runtime_int[i].begin(),
                               host_runtime_int[i].end(),
                               soa.GetIntData(NAI+i).begin() + old_size);
            }

            Gpu::streamSynchronize();
        }

        RedistributeLocal();
    }

    void moveParticles (const IntVect& move_dir, int do_random)
    {
        BL_PROFILE("TestParticleContainer::moveParticles");

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            const auto dx = Geom(lev).CellSizeArray();
            auto& plev  = GetParticles(lev);

            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                auto& ptile = plev[std::make_pair(gid, tid)];
                auto& aos   = ptile.GetArrayOfStructs();
                ParticleType* pstruct = aos.data();
                const size_t np = aos.numParticles();

                if (do_random == 0)
                {
                    amrex::ParallelFor(np,
                    [=] AMREX_GPU_DEVICE (size_t i) noexcept
                    {
                        ParticleType& p = pstruct[i];
                        p.pos(0) += static_cast<ParticleReal> (move_dir[0]*dx[0]);
#if AMREX_SPACEDIM > 1
                        p.pos(1) += static_cast<ParticleReal> (move_dir[1]*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                        p.pos(2) += static_cast<ParticleReal> (move_dir[2]*dx[2]);
#endif
                    });
                }
                else
                {
                    amrex::ParallelForRNG(np,
                    [=] AMREX_GPU_DEVICE (size_t i, RandomEngine const& engine) noexcept
                    {
                        ParticleType& p = pstruct[i];

                        p.pos(0) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*move_dir[0]*dx[0]);
#if AMREX_SPACEDIM > 1
                        p.pos(1) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*move_dir[1]*dx[1]);
#endif
#if AMREX_SPACEDIM > 2
                        p.pos(2) += static_cast<ParticleReal> ((2*amrex::Random(engine)-1)*move_dir[2]*dx[2]);
#endif
                    });
                }
            }
        }
    }

    void checkAnswer () const
    {
        BL_PROFILE("TestParticleContainer::checkAnswer");

        AMREX_ALWAYS_ASSERT(OK());

        int num_rr = NumRuntimeRealComps();
        int num_ii = NumRuntimeIntComps();

        for (int lev = 0; lev <= finestLevel(); ++lev)
        {
            const auto& plev  = GetParticles(lev);

            for(MFIter mfi = MakeMFIter(lev); mfi.isValid(); ++mfi)
            {
                int gid = mfi.index();
                int tid = mfi.LocalTileIndex();
                const auto& ptile = plev.at(std::make_pair(gid, tid));
                const auto& ptd = ptile.getConstParticleTileData();
                const size_t np = ptile.numParticles();

                AMREX_FOR_1D ( np, i,
                {
                    for (int j = 0; j < NSR; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].rdata(j) == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NSI; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_aos[i].idata(j) == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NAR; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_rdata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < NAI; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_idata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < num_rr; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_rdata[j][i] == ptd.m_aos[i].id());
                    }
                    for (int j = 0; j < num_ii; ++j)
                    {
                        AMREX_ALWAYS_ASSERT(ptd.m_runtime_idata[j][i] == ptd.m_aos[i].id());
                    }
                });
            }
        }
    }
};

int main(int argc, char* argv[])
{

    MPI_Comm comm = amrex::MPMD::Initialize(argc, argv);
    amrex::Initialize(argc,argv,true,comm);
    {
        amrex::Print() << "Hello world from AMReX version " << amrex::Version() << "\n";
        // how many grid cells in each direction over the problem domain
        int n_cell = 32;
        // how many grid cells are allowed in each direction over each box
        int max_grid_size = 16;
        // integer vector indicating the lower coordindate bounds
        amrex::IntVect dom_lo(0,0,0);
        // integer vector indicating the upper coordindate bounds
        amrex::IntVect dom_hi(n_cell-1, n_cell-1, n_cell-1);
        // box containing the coordinates of this domain
        amrex::Box domain(dom_lo, dom_hi);
        // will contain a list of boxes describing the problem domain
        amrex::BoxArray ba(domain);
        // chop the single grid into many small boxes
        ba.maxSize(max_grid_size);
        // Distribution Mapping
        amrex::DistributionMapping dm(ba);
        //Geometry -- Physical Properties for data on our domain
        amrex::RealBox real_box ({0., 0., 0.}, {1. , 1., 1.});
        amrex::Geometry geom(domain, &real_box);
        // Create an MPMD Copier
        auto copr = amrex::MPMD::Copier(ba,dm,false);

	// Will be sending information from this app

	// Create new DistributionMap for this new BoxArray
        amrex::DistributionMapping dm_1(copr.Other_boxArray());
	// Send the new DistributionMap information to other app
	auto copr_1 = amrex::MPMD::Copier(copr.Other_boxArray(),dm_1,true);

	// Create ParticleContainer and populate with Data
        TestParticleContainer pc(geom, dm, ba);
	// Initialize 
        IntVect nppc(10);
        pc.InitParticles(nppc);

	// Create a new ParticleContainer with the other_boxArray
        TestParticleContainer opc_1(geom, dm_1, copr.Other_boxArray());
	// Copy to the new particle container
	opc_1.copyParticles(pc,false);
	// Change the DistributionMapping of opc_1
	opc_1.SetParticleDistributionMap(0,copr.Other_DistributionMap());
	// Change the MPI communicator
	MPI_Comm global_comm = MPI_COMM_WORLD;
	ParallelContext::push(global_comm);
	// Call Redistribute to send data to other app?
	opc_1.Redistribute();
	ParallelContext::pop();
    }
    amrex::Finalize();

    amrex::MPMD::Finalize();

}

