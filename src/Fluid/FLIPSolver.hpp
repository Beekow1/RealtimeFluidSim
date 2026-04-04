class FLIPSolver {
public:
    FLIPSolver(int nx, int ny, int nz, float h)
        : grid(nx, ny, nz, h),
        cellType(nx* ny* nz, AIR) {}

    void step(float dt);

private:
    MACGrid grid;
    std::vector<Particle> particles;
    std::vector<CellType> cellType;

    void advectParticles(float dt);
    void markFluidCells();
    void particlesToGrid();
    void addGravity(float dt);
    void solvePressure(float dt);
    void applyBoundaryConditions();
    void gridToParticles(const MACGrid& oldGrid, float dt);

    Vec3 sampleMAC(const MACGrid& g, const Vec3& x) const;
};