      subroutine cavovo
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
c
c     *** calculate void volumes
c
c     *** initialize void volumes
c
      do 10 k=1,nvrm
   10 vol(k)=0.0
c
c     *** compute void region volumes
c
      do 30 j=2,jm1
      do 30 i=2,im1
      inf=nf(i,j)
      if (inf.eq.0.or.beta(i,j).lt.0.0) go to 30
      vola=(1.0-f(i,j))*delx(i)*dely(j)*(1.0-cyl+cyl*2.0*3.14159*xi(i))
      if (inf.gt.5) go to 20
      infr=nf(i+1,j)
      inft=nf(i,j+1)
      infl=nf(i-1,j)
      infb=nf(i,j-1)
      inf=max0(infr,inft,infl,infb)
   20 vol(inf)=vol(inf)+vola
   30 continue
      return
      end
