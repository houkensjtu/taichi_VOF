C
C                           PROGRAM SOLAVOF
C
C      program solavof (input,tape5=input,output,tape6=output,tape7,tape8
C     1 )
c
c     *** sola-vof  volume of fluid method ***
c
c     *** list of primary variables ***
c
c     *** input parameters (namelist /xput/)
c
c     alpha     controls amount of donor cell fluxing (=1.0 for full
c               donor cell differencing, =0.0 for central differencing)
c     autot     automatic time step flag (=1.0 for automatic delt
c               adjustment, =0.0 for constant delt)
c     cangle    contact angle, in degres, between fluid and wall
c     csq       material sound speed squared (=-1.0 for
c               incompressible material)
c     delt      time step
c     epsi      pressure iteration convergence criterion
c     flht      fluid height, in y-direction
c     gx        body acceleration in positive x-direction
c     gy        body acceleration in positive y-direction
c     icyl      mesh geometry indicator (=1 for cylindrical coordinates,
c               =0 for plane coordinates)
c     imovy     movie indicator (=1 for movie film output, =0 for
c               other film output)
c     isurf10   surface tension indicator (=1 for surface tension,
c               =0 for no surface tension)
c     isymplt   symmetry plot indicator (=1 for symmetry plot,
c               =0 for no symmetry plot)
c     nmat      number of materials
c     npx       number of particles in x-direction in rectangular setup
c     npy       number of particles in y-direction in rectangular setup
c     nu        coefficient of kinematic viscosity
c     omg       over-relaxation factor used in pressure iteration
c     pltdt     time increment between plots and/or prints to be output
c               on film
c     prtdt     time increment between prints on paper
c     rhof      fluid density (for f=1.0 region)
c     rhofc     fluid density in complement of f region
c     sigma     surface tension coefficient
c     twfin     problem time to end calculation
c     ui        x-direction velocity used for initializing mesh
c     vi        y-direction velocity used for initializing mesh
c     velmx     maximum velocity expected in problem, used to scale
c               velocity vectors
c     wb        indicator for boundary condition to be used along the
c               bottom of the mesh (=1 for rigid free-slip wall,
c               =2 for rigid no-slip wall, =3 for continuative
c               boundary, =4 for periodic boundary, =5 for constant
c               pressure boundary)
c     wl        indicator for boundary condition along left side of
c               mesh (see wb)
c     wr        indicator for boundary condition along right side of
c               mesh (see wb)
c     wt        indicator for boundary condition along top of mesh
c               (see wb)
c     xpl       location of left side of rectangular particle region
c     xpr       location of right side of rectangular particle region
c     ypb       location of bottom of rectangular particle region
c     ypt       location of top of rectangular particle region
c
c     *** mesh setup input (namelist /mshset/)
c
c     dxmn(n)     minimum space increment in x-direction in submesh n
c     dymn(n)     minimum space increment in y-direction in submesh n
c     nkx         number of submesh regions in x-direction
c     nxl(n)      number of cells between locations xl(n) and xc(n) in
c                 submesh n
c     nxr(n)      number of cells between locations xc(n) and xl(n+1) in
c                 submesh n
c     nyl(n)      number of cells between locations yl(n) and yc(n) in
c                 submesh n
c     nyr(n)      number of cells between locations yc(n) and yl(n+1) in
c                 submesh n
c     xc(n)       x-coordinate of the convergence point in submesh n
c     xl(n)       location of the left edge of submesh n (nkx+1 values
c                 of xl(n) are necessary because the right edge (xr) of
c                 submesh n is determined by the left edge of
c                 submesh n+1)
c     yc(n)       y-coordinate of the convergence point in submesh n
c     yl(n)       location of the bottom of submesh n (nky+1 values of
c                 yl(n) are necessary because the top edge (yr) of
c                 submesh n is determined by the bottom edge of
c                 submesh n+1)
c
c     *** variables listed in common (excluding input parameters)
c
c     cycle     calculational time cycle
c     cyl       mesh geometry indicator (= icyl)
c     dtsft     maximum delt value allowed by the surface tension forces
c               stability criterion (delt is automatically adjusted)
c     dtvis     maximum delt value allowed by the viscous forces
c               stability criterion (delt is automatically adjusted)
c     emf       small value, typically 1.0e-06, used to negate round-off
c               error effects when testing f=1.0 or f=0.0
c     emf1      =1.0-emf
c     em6       =1.0e-06
c     em10      =1.0e-10
c     ep10      =1.0e+10
c     flg       pressure iteration convergence test indicator (=0.0 when
c               the convergence test is satisfied, =1.0 when the
c               convergence test is not satisfied)
c     flgc      volume of fluid function convection limit indicator
c               (delt reduced and cycle started over if limit
c               is exceeded)
c     fnoc      pressure convergence failure indicator (=1.0,
c               convergence failed and delt is reduced, =0.0 otherwise)
c     ibar      number of real cells in x-direction (excludes ficticious
c               cells)
c     ibar2     =ibar+2, specified in parameter statement
c               (=ibar+3, if periodic in x-direction)
c     imax      total number of mesh cells in x-direction (=ibar+2)
c               (=ibar+3, if periodic in x-direction)
c     im1       value of the index i at the last real cell in the
c               x-direction (=imax-1)
c     im2       value of the index i at the next to the last real cell
c               in the x-direction (=imain the x-2)
c     ipl       leftmost pressure iteration index in x-direction
c               (=3 for constant pressure boundary condition, =2 for
c               all other cases)
c     ipr       rightmost pressure iteration index in x-direction
c               (=im2 for constant pressure boundary condition, =im1 for
c               all other cases)
c     iter      pressure iteration counter
c     jbar      number of real cells in y-direction (excludes ficticious
c               cells)
c     jbar2     =jbar+2, specified in parameter statement
c               (=jbar+3, if periodic in y-direction)
c     jmax      total number of mesh cells in y-direction (=jbar+2)
c               (=jbar+3, if periodic in y-direction)
c     jm1       value of the index j at the last real cell in the
c               y-direction (=jmax-1)
c     jm2       value of the index j at the next to the last real cell
c               in the y-direction (=jmax-2)
c     jpb       bottom pressure iteration index in y-direction
c               (=3 for constant pressure boundary condition, =2 for
c               all other cases)
c     jpt       top pressure iteration index in y-direction
c               (=jm2 for constant pressure boundary condition, =jm1 for
c               all other cases)
c     nflgc     number of cycles the volume of fluid function convection
c               limit (flgc) is exceeded
c     nocon     number of cycles pressure convegence has failed
c     np        total number of particles computed to be in mesh
c     nprts     number of particles in mesh, specfied in parameter state
c               (used to set array size - must be > 0)
c     nreg      number of void regions generated in calculation
c     nvor      maximum number of void regions allowed, specified in
c               parameter statement
c     nvrm      number of void regions
c     meshx     number of submesh regions in x-direction, specified
c               in parameter statement
c     meshy     number of submesh regions in y-direction, specified
c               in parameter statement
c     pi        =3.141592654
c     rcsq      reciprocal of csq
c     rhod      difference in fluid densities (=rhof-rhofc)
c     rpd       degrees to radians conversion factor
c     sf        plot scaling factor
c     t         problem time
c     tangle    tangent of contact angle, cangle
c     vchgt     accumulated fluid volume change
c     velmx1    velmx normalized to minimum mesh cell dimension
c     xmax      location of right-hand side of mesh
c     xmin      location of left-hand side of mesh
c     xshft     computed shift along the plotting abscissa to center
c               the plot frame on film
c     ymax      location of the top of the mesh
c     ymin      location of the bottom of the mesh
c     yshft     computed shift along the plotting ordinate to center the
c               plot frame on film
c
c     *** arrays in common (excluding mesh setup parameters)
c
c     acom(1)     first word in common
c     beta(i,j)   pressure iteration relaxation factor in cell (i,j)
c     delx(i)     mesh spacing of the i-th cell along the x-axis
c     dely(j)     mesh spacing of the j-th cell along the y-axis
c     f(i,j)      volume of fluid per unit volume of cell (i,j) at time
c                 level n+1
c     fn(i,j)     volume of fluid per unit volume of cell (i,j) at time
c                 level n
c     ip(k)       cell index for particle k along x-axis
c     jp(k)       cell index for particle k along y-axis
c     name(10)    problem identification line
c     nf(i,j)     flag of surface cell (i,j) indicating the location
c                 of its neighboring pressure interpolation cell
c     nr(k)       label of void region, k > 5
c     p(i,j)      pressure in cell (i,j) at time level n+1
c     peta(i,j)   pressure interpolation factor for cell (i,j)
c     pn(i,j)     pressure in cell (i,j) at time level n
c     pr(k)       pressure in void region nr(k)
c     ps(i,j)     surface pressure in cell (i,j) computed from surface
c                 tension forces
c     rdx(i)      reciprocal of delx(i)
c     rdy(j)      reciprocal of dely(j)
c     rx(i)       reciprocal of x(i)
c     rxi(i)      reciprocal of xi(i)
c     ryj(j)      reciprocal of yj(j)
c     tanth(i,j)  slope of fluid surface in cell (i,j)
c     u(i,j)      x-direction velocity component in cell (i,j) at time
c                 level n+1
c     un(i,j)     x-direction velocity component in cell (i,j) at time
c                 level n
c     v(i,j)      y-direction velocity component in cell (i,j) at time
c                 level n+1
c     vn(i,j)     y-direction velocity component in cell (i,j) at time
c                 level n
c     vol(k)      volume of void region nr(k)
c     x(i)        location of the right-hand boundary of the i-th cell
c                 along the x-axis
c     xi(i)       location of the center of the i-th cell along the
c                 x-axis
c     xp(k)       x-coordinate of particle k
c     y(j)        location of the top boundary of the j-th cell along th
c                 y-axis
c     yj(j)       location of the center of the j-th cell along the
c                 y-axis
c     yp(k)       y-coordinate of particle k
c     zcom(1)     last word in common
*     ----- begin comdeck common1    -----
      parameter (ibar2=22, jbar2=10, nprts=1, meshx=1, meshy=1, nvor=10)
      parameter (mshx=meshx+1, mshy=meshy+1)
c
      real nu, normx, normy
      integer cycle, wl, wr, wt, wb
c
      common /fv/ acom(1), un(ibar2,jbar2), vn(ibar2,jbar2), pn(ibar2
     1 ,jbar2), fn(ibar2,jbar2), u(ibar2,jbar2), v(ibar2,jbar2), p(ibar2
     2 ,jbar2), f(ibar2,jbar2), peta(ibar2,jbar2), beta(ibar2,jbar2), nf
     3 (ibar2,jbar2), tanth(ibar2,jbar2), ps(ibar2,jbar2)
c
      common /me/ x(ibar2), xi(ibar2), rxi(ibar2), delx(ibar2), rdx
     1 (ibar2), rx(ibar2), y(jbar2), yj(jbar2), ryj(jbar2), dely(jbar2),
     2 rdy(jbar2), xl(mshx), xc(meshx), dxmn(meshx), nxl(meshx), nxr
     3 (meshx), yl(mshy), yc(meshy), dymn(meshy), nyl(meshy), nyr(meshy)
      common /pv/ xp(nprts), yp(nprts), ip(nprts), jp(nprts), nr(nvor),
     1 pr(nvor), vol(nvor), name(10)
c
      common /iv/ ibar, jbar, imax, jmax, im1, jm1, im2, jm2, nkx, nky,
     1 cycle, delt, t, autot, prtdt, twprt, pltdt, twplt, twfin, flht,
     2 nu, csq, rcsq, nmat, rhof, rhofc, rhod, nvrm, nreg,vchgt,rdtexp,
     3 isurf10, sigma, cangle, tanca, icyl, cyl, gx, gy, ui, vi, omg,
     4 alpha, wl, wr, wb, wt, np, iter, epsi, flg, flgc, fnoc, nocon,
     5 nflgc, isymplt, imovy, velmx, velmx1, xshft, yshft, xmin, xmax,
     6 ymin, ymax, sf, xpl, xpr, ypb, ypt, npx, npy, ipl, ipr, jpb, jpt,
     7 dtvis,dtsft
c
      common /const/ emf, emf1, em6, em10, ep10, pi, rpd
c
      common /last/ zcom
*     ----- end comdeck common1    -----
      namelist /xput/ delt,nu,icyl,epsi,gx,gy,ui,vi,velmx,twfin,prtdt
     1 ,pltdt,omg,alpha,wl,wr,wt,wb,imovy,autot,flht,isymplt,sigma
     2 ,isurf10,cangle,csq,nmat,rhof,rhofc,xpl,xpr,ypb,ypt,npx,npy
      namelist /mshset/ nkx,xl,xc,nxl,nxr,dxmn,nky,yl,yc,nyl,nyr,dymn
c
      data emf /1.0e-06/, em6 /1.0e-06/, em10 /1.0e-10/
      data ep10 /1.0e+10/
      data pi /3.141592654/, rpd /0.0174532925/
c
c     *** default input data
c     *** note    user must supply the following regardless
c                      of defaults: delt,twfin,prtdt,pltdt
c
      data nu /0.0/, icyl /0/, epsi /1.0e-03/, gx /0.0/, gy /0.0/, ui /0
     1 .0/, vi /0.0/, velmx /1.0/, imovy /0/, omg /1.7/, alpha /1.0/, wl
     2 /1/, wr /1/, wt /1/, wb /1/, csq /-1.0/, autot /1.0/, isymplt /0/
     3 , isurf10 /0/, sigma /0.0/, cangle /90.0/, nmat /1/, rhof /1.0/,
     4 rhofc /1.0/, flht /0.0/, xpl /0.0/, ypb /0.0/, xpr /0.0/, ypt /0.
     5 0/, npx /0/, npy /0/
C
      delt=0.0001
      twfin = 0.01
      prtdt=0.0001
      pltdt=0.01
c
c     *** setup film initialization to system
c     *** note    filmset is system dependant
c
C      call filmset
c
c     *** read problem title (name)
c
      read (5,110) name
c
c     *** read initial input data
c
      read (5,xput)
c
c     *** read input parameters for variable mesh
c
      read (5,mshset)
c
c     *** calculate variable mesh data
c
      call meshset
c
c     *** print initial input data
c
      call prtplt (1)
c
c     *** set initial conditions
c
      call setup
c
c     *** set initial boundary conditions
c
      call bc
c
      go to 20
c
c     *** start time cycle
c
   10 continue
      iter=0
      flg=1.0
      fnoc=0.0
c
c     *** explicitly approximate new time-level velocities
c
      call tilde
c
      if (nmat.eq.2.and.isurf10.eq.1) call tms10
c
c     *** set boundary conditions
c
      call bc
c
c     *** iteratively adjust cell pressure and velocity
c
      call pressit
c
      if (t.gt.ep10) go to 30
c
c     *** update fluid configuration
c
   20 call vfconv
c
      if (flgc.gt.0.5) go to 90
c
c     *** set boundary conditions
c
      call bc
c
c     *** move marker particles
c
      call parmov
c
c     *** determine pressure interpolation factor and neighbor
c     *** also determine surface tension pressures and
c     *** wall adhesion effects in surface cells
c
      call petacal
c
c     *** print time and cycle data on paper and/or film
c
   30 call prtplt (2)
c
      if (cycle.le.0) go to 40
      if (t+em6.lt.twplt) go to 50
      twplt=twplt+pltdt
   40 continue
c
c     *** print field variable data on film
c
      call prtplt (3)
c
c     *** plot velocity vector, free surface, mesh,
c     *** and marker particle on film
c
C      call draw
c
   50 continue
      if (cycle.le.0) go to 60
      if (t+em6.lt.twprt) go to 70
      twprt=twprt+prtdt
   60 continue
c
c     *** print field variable data on paper
c
      call prtplt (4)
c
   70 continue
c
c     *** set the advance time arrays into the time-n arrays
c
      do 80 i=1,imax
      do 80 j=1,jmax
      un(i,j)=u(i,j)
      vn(i,j)=v(i,j)
      u(i,j)=0.0
      v(i,j)=0.0
      pn(i,j)=p(i,j)
      fn(i,j)=f(i,j)
   80 continue
      nregn=nreg
c
c     *** adjust delt
c
   90 call deltadj
c
c     *** advance time
c
      t=t+delt
      if (t.gt.twfin) go to 100
c
c     *** advance cycle
c
      cycle=cycle+1
      if (nflgc.ge.25.or.nocon.ge.25) t=ep10
      go to 10
c
  100 call exit
c
  110 format (10a8)
      end
