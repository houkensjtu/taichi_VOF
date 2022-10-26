      subroutine tms10
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
c     *** two material surface tension
c
c     *** note: this routine introduces some numerical noise
c     ***       and may be replaced in the future
c
      do 30 i=2,im1
      do 30 j=2,jm1
      if (nf(i,j).eq.0.or.nf(i,j).ge.5.or.beta(i,j).lt.0.0) go to 30
      whtl=0.5
      whtr=0.5
      whtt=0.5
      whtb=0.5
      if (nf(i,j).gt.2) go to 10
      whtl=1.0-f(i,j)
      if (nf(i,j).eq.2) whtl=1.0-whtl
      whtr=1.0-whtl
      stfx=ps(i,j)*dely(j)
      if (nf(i,j).eq.1) stfx=-stfx
      stfy=stfx*tanth(i,j)
      go to 20
   10 whtb=1.0-f(i,j)
      if (nf(i,j).eq.4) whtb=1.0-whtb
      whtt=1.0-whtb
      stfy=ps(i,j)*delx(i)
      if (nf(i,j).eq.3) stfy=-stfy
      stfx=-stfy*tanth(i,j)
   20 continue
      rhoxr=(rhofc+rhod*f(i,j))*delx(i+1)+(rhofc+rhod*f(i+1,j))*delx(i)
      u(i,j)=u(i,j)+2.0*delt*whtr*stfx/(rhoxr*dely(j))
      rhoxl=(rhofc+rhod*f(i-1,j))*delx(i)+(rhofc+rhod*f(i,j))*delx(i-1)
      u(i-1,j)=u(i-1,j)+2.0*delt*whtl*stfx/(rhoxl*dely(j))
      rhoyt=(rhofc+rhod*f(i,j))*dely(j+1)+(rhofc+rhod*f(i,j+1))*dely(j)
      v(i,j)=v(i,j)+2.0*delt*whtt*stfy/(rhoyt*delx(i))
      rhoyb=(rhofc+rhod*f(i,j-1))*dely(j)+(rhofc+rhod*f(i,j))*dely(j-1)
      v(i,j-1)=v(i,j-1)+2.0*delt*whtb*stfy/(rhoyb*delx(i))
   30 continue
      return
      end
