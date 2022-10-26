      subroutine meshset
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
c     *** mesh setup  (generation)
c
      i=1
      j=1
      x(1)=xl(1)
      y(1)=yl(1)
      do 30 k=1,nkx
      dxml=(xc(k)-xl(k))/nxl(k)
      dxmr=(xl(k+1)-xc(k))/nxr(k)
      dxmn1=dxmn(k)
      nt=nxl(k)
      tn=nt
      tn=amax1(tn,1.0+em6)
      dxmn(k)=amin1(dxmn1,dxml)
      cmc=(xc(k)-xl(k)-tn*dxmn(k))*tn/(tn-1.0)
      if (nt.eq.1) cmc=0.0
      bmc=xc(k)-xl(k)-cmc
      do 10 l=1,nt
      i=i+1
      rln=(float(l)-tn)/tn
   10 x(i)=xc(k)+bmc*rln-cmc*rln*rln
      nt=nxr(k)
      tn=nt
      tn=amax1(tn,1.0+em6)
      dxmn(k)=amin1(dxmn1,dxmr)
      cmc=(xl(k+1)-xc(k)-tn*dxmn(k))*tn/(tn-1.0)
      if (nt.eq.1) cmc=0.0
      bmc=xl(k+1)-xc(k)-cmc
      do 20 l=1,nt
      i=i+1
      rln=float(l)/tn
   20 x(i)=xc(k)+bmc*rln+cmc*rln*rln
   30 continue
      if (wr.ne.4) go to 40
      i=i+1
      x(i)=x(i-1)+x(2)-x(1)
   40 continue
      do 70 k=1,nky
      dyml=(yc(k)-yl(k))/nyl(k)
      dymr=(yl(k+1)-yc(k))/nyr(k)
      dymn1=dymn(k)
      nt=nyl(k)
      tn=nt
      tn=amax1(tn,1.0+em6)
      dymn(k)=amin1(dymn1,dyml)
      cmc=(yc(k)-yl(k)-tn*dymn(k))*tn/(tn-1.0)
      if (nt.eq.1) cmc=0.0
      bmc=yc(k)-yl(k)-cmc
      do 50 l=1,nt
      j=j+1
      rln=(float(l)-tn)/tn
   50 y(j)=yc(k)+bmc*rln-cmc*rln*rln
      nt=nyr(k)
      tn=nt
      tn=amax1(tn,1.0+em6)
      dymn(k)=amin1(dymn1,dymr)
      cmc=(yl(k+1)-yc(k)-tn*dymn(k))*tn/(tn-1.0)
      if (nt.eq.1) cmc=0.0
      bmc=yl(k+1)-yc(k)-cmc
      do 60 l=1,nt
      j=j+1
      rln=float(l)/tn
   60 y(j)=yc(k)+bmc*rln+cmc*rln*rln
   70 continue
      if (wt.ne.4) go to 80
      j=j+1
      y(j)=y(j-1)+y(2)-y(1)
   80 continue
      numx=i
      numy=j
      numxm1=numx-1
      numym1=numy-1
      numxp1=numx+1
      numyp1=numy+1
      ibar=numx-1
      jbar=numy-1
      imax=ibar+2
      jmax=jbar+2
      im1=imax-1
      jm1=jmax-1
      im2=imax-2
      jm2=jmax-2
c
c     *** calculate values needed for variable mesh
c
      do 100 i=1,numx
      if (x(i).eq.0.0) go to 90
      rx(i)=1.0/x(i)
      go to 100
   90 rx(i)=0.0
  100 continue
      do 110 i=2,numx
      xi(i)=0.5*(x(i-1)+x(i))
      delx(i)=x(i)-x(i-1)
      rxi(i)=1.0/xi(i)
  110 rdx(i)=1.0/delx(i)
      delx(1)=delx(2)
      xi(1)=xi(2)-delx(2)
      rxi(1)=1.0/xi(1)
      rdx(1)=1.0/delx(1)
      delxa=delx(numx)
      if(wr.eq.4) delxa=delx(3)
      delx(numxp1)=delxa
      xi(numxp1)=xi(numx)+delxa
      x(numxp1)=xi(numxp1)+0.5*delx(numxp1)
      rxi(numxp1)=1.0/xi(numxp1)
      rdx(numxp1)=1.0/delx(numxp1)
      do 120 i=2,numy
      yj(i)=0.5*(y(i-1)+y(i))
      ryj(i)=1.0/yj(i)
      dely(i)=y(i)-y(i-1)
      rdy(i)=1.0/dely(i)
  120 continue
      dely(1)=dely(2)
      rdy(1)=1.0/dely(1)
      yj(1)=yj(2)-dely(2)
      ryj(1)=1.0/yj(1)
      delya=dely(numy)
      if(wt.eq.4) delya=dely(3)
      dely(numyp1)=delya
      yj(numyp1)=yj(numy)+delya
      y(numyp1)=yj(numyp1)+0.5*dely(numyp1)
      ryj(numyp1)=1.0/yj(numyp1)
      rdy(numyp1)=1.0/dely(numyp1)
      write (6,190)
      do 130 i=1,numxp1
      write (6,200) i,x(i),i,rx(i),i,delx(i),i,rdx(i),i,xi(i),i,rxi(i)
  130 continue
      write (6,190)
      do 140 i=1,numyp1
      write (6,210) i,y(i),i,dely(i),i,rdy(i),i,yj(i),i,ryj(i)
  140 continue
      if (imovy.eq.1) go to 170
      write (12,190)
      do 150 i=1,numxp1
      write (12,200) i,x(i),i,rx(i),i,delx(i),i,rdx(i),i,xi(i),i,rxi(i)
  150 continue
      write (12,190)
      do 160 i=1,numyp1
      write (12,210) i,y(i),i,dely(i),i,rdy(i),i,yj(i),i,ryj(i)
  160 continue
  170 continue
c
c     *** test array size
c
      if (imax.le.ibar2.and.jmax.le.jbar2) go to 180
      write (6,220)
c
      call exit
c
  180 continue
      return
c
  190 format (1h1)
  200 format (1x,2hx(,i2,2h)=,1pe12.5,2x,3hrx(,i2,2h)=,1pe12.5,2x,5hdelx
     1(,i2,2h)=,1pe12.5,1x,4hrdx(,i2,2h)=,1pe12.5,2x,3hxi(,i2,2h)=,1pe12
     2 .5,2x,4hrxi(,i2,2h)=,1pe12.5)
  210 format (1x,2hy(,i2,2h)=,1pe12.5,3x,5hdely(,i2,2h)=,1pe12.5,3x,4hrd
     1y(,i2,2h)=,1pe12.5,3x,3hyj(,i2,2h)=,1pe12.5,3x,4hryj(,i2,2h)=,1pe1
     2 2.5)
  220 format (41h  mesh size greater than array dimensions)
      end
