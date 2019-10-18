C Subset of the original Fortran functions for Tsyganenko T96 model
C Linked using Numpy's F2PY tools in main program for performance reasons
C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
C THIS CODE RETURNS THE SHIELDING FIELD REPRESENTED BY  2x3x3=18 "CARTESIAN"
C    HARMONICS
C
      SUBROUTINE  SHLCAR3X3(A,X,Y,Z,SPS,HX,HY,HZ)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C  The 36 coefficients enter in pairs in the amplitudes of the "cartesian"
c    harmonics (A(1)-A(36).
c  The 12 nonlinear parameters (A(37)-A(48) are the scales Pi,Ri,Qi,and Si
C   entering the arguments of exponents, sines, and cosines in each of the
C   18 "Cartesian" harmonics
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      IMPLICIT REAL * 8  (A - H, O - Z)
      DIMENSION A(48)
C
C
C
      CPS=DSQRT(1.D0-SPS**2)
      S3PS=4.D0*CPS**2-1.D0   !  THIS IS SIN(3*PS)/SIN(PS)
C
      HX=0.D0
      HY=0.D0
      HZ=0.D0
      L=0
C
      DO 1 M=1,2     !    M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY)
C                           AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY)
            DO 2 I=1,3
                  P=A(36+I)
                  Q=A(42+I)
                  CYPI=DCOS(Y/P)
                  CYQI=DCOS(Y/Q)
                  SYPI=DSIN(Y/P)
                  SYQI=DSIN(Y/Q)
C
            DO 3 K=1,3
                  R=A(39+K)
                  S=A(45+K)
                  SZRK=DSIN(Z/R)
                  CZSK=DCOS(Z/S)
                  CZRK=DCOS(Z/R)
                  SZSK=DSIN(Z/S)
                  SQPR=DSQRT(1.D0/P**2+1.D0/R**2)
                  SQQS=DSQRT(1.D0/Q**2+1.D0/S**2)
                  EPR=DEXP(X*SQPR)
                  EQS=DEXP(X*SQQS)
C
                  DO 4 N=1,2  ! N=1 IS FOR THE FIRST PART OF EACH COEFFICIENT
C                                  AND N=2 IS FOR THE SECOND ONE
C
                    L=L+1
                     IF (M.EQ.1) THEN
                       IF (N.EQ.1) THEN
                         DX=-SQPR*EPR*CYPI*SZRK
                         DY=EPR/P*SYPI*SZRK
                         DZ=-EPR/R*CYPI*CZRK
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ELSE
                         DX=DX*CPS
                         DY=DY*CPS
                         DZ=DZ*CPS
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ENDIF
                     ELSE
                       IF (N.EQ.1) THEN
                         DX=-SPS*SQQS*EQS*CYQI*CZSK
                         DY=SPS*EQS/Q*SYQI*CZSK
                         DZ=SPS*EQS/S*CYQI*SZSK
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ELSE
                         DX=DX*S3PS
                         DY=DY*S3PS
                         DZ=DZ*S3PS
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                       ENDIF
                 ENDIF
c
  4   CONTINUE
  3   CONTINUE
  2   CONTINUE
  1   CONTINUE
C
         RETURN
          END

C
C@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

C$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
C
      SUBROUTINE  BIRK1SHLD(PS,X,Y,Z,BX,BY,BZ)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The 64 linear parameters are amplitudes of the "box" harmonics.
c The 16 nonlinear parameters are the scales Pi, and Qk entering the arguments
C  of sines/cosines and exponents in each of  32 cartesian harmonics
c  N.A. Tsyganenko, Spring 1994, adjusted for the Birkeland field Aug.22, 1995
c    Revised  June 12, 1996.
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
      IMPLICIT  REAL * 8  (A - H, O - Z)
C
      DIMENSION A(80)
      DIMENSION P1(4),R1(4),Q1(4),S1(4),RP(4),RR(4),RQ(4),RS(4)
C
      EQUIVALENCE (P1(1),A(65)),(R1(1),A(69)),(Q1(1),A(73)),
     * (S1(1),A(77))
C
      DATA A/1.174198045,-1.463820502,4.840161537,-3.674506864,
     * 82.18368896,-94.94071588,-4122.331796,4670.278676,-21.54975037,
     * 26.72661293,-72.81365728,44.09887902,40.08073706,-51.23563510,
     * 1955.348537,-1940.971550,794.0496433,-982.2441344,1889.837171,
     * -558.9779727,-1260.543238,1260.063802,-293.5942373,344.7250789,
     * -773.7002492,957.0094135,-1824.143669,520.7994379,1192.484774,
     * -1192.184565,89.15537624,-98.52042999,-0.8168777675E-01,
     * 0.4255969908E-01,0.3155237661,-0.3841755213,2.494553332,
     * -0.6571440817E-01,-2.765661310,0.4331001908,0.1099181537,
     * -0.6154126980E-01,-0.3258649260,0.6698439193,-5.542735524,
     * 0.1604203535,5.854456934,-0.8323632049,3.732608869,-3.130002153,
     * 107.0972607,-32.28483411,-115.2389298,54.45064360,-0.5826853320,
     * -3.582482231,-4.046544561,3.311978102,-104.0839563,30.26401293,
     * 97.29109008,-50.62370872,-296.3734955,127.7872523,5.303648988,
     * 10.40368955,69.65230348,466.5099509,1.645049286,3.825838190,
     * 11.66675599,558.9781177,1.826531343,2.066018073,25.40971369,
     * 990.2795225,2.319489258,4.555148484,9.691185703,591.8280358/
C
         BX=0.D0
         BY=0.D0
         BZ=0.D0
         CPS=DCOS(PS)
         SPS=DSIN(PS)
         S3PS=4.D0*CPS**2-1.D0
C
         DO 11 I=1,4
          RP(I)=1.D0/P1(I)
          RR(I)=1.D0/R1(I)
          RQ(I)=1.D0/Q1(I)
 11       RS(I)=1.D0/S1(I)
C
          L=0
C
           DO 1 M=1,2     !    M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY)
C                           AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY)
             DO 2 I=1,4
                  CYPI=DCOS(Y*RP(I))
                  CYQI=DCOS(Y*RQ(I))
                  SYPI=DSIN(Y*RP(I))
                  SYQI=DSIN(Y*RQ(I))
C
                DO 3 K=1,4
                   SZRK=DSIN(Z*RR(K))
                   CZSK=DCOS(Z*RS(K))
                   CZRK=DCOS(Z*RR(K))
                   SZSK=DSIN(Z*RS(K))
                     SQPR=DSQRT(RP(I)**2+RR(K)**2)
                     SQQS=DSQRT(RQ(I)**2+RS(K)**2)
                        EPR=DEXP(X*SQPR)
                        EQS=DEXP(X*SQQS)
C
                    DO 4 N=1,2  ! N=1 IS FOR THE FIRST PART OF EACH COEFFICIENT
C                                  AND N=2 IS FOR THE SECOND ONE
                     IF (M.EQ.1) THEN
                       IF (N.EQ.1) THEN
                         HX=-SQPR*EPR*CYPI*SZRK
                         HY=RP(I)*EPR*SYPI*SZRK
                         HZ=-RR(K)*EPR*CYPI*CZRK
                                   ELSE
                         HX=HX*CPS
                         HY=HY*CPS
                         HZ=HZ*CPS
                                   ENDIF
                     ELSE
                       IF (N.EQ.1) THEN
                         HX=-SPS*SQQS*EQS*CYQI*CZSK
                         HY=SPS*RQ(I)*EQS*SYQI*CZSK
                         HZ=SPS*RS(K)*EQS*CYQI*SZSK
                                   ELSE
                         HX=HX*S3PS
                         HY=HY*S3PS
                         HZ=HZ*S3PS
                       ENDIF
                 ENDIF
       L=L+1
c
       BX=BX+A(L)*HX
       BY=BY+A(L)*HY
  4    BZ=BZ+A(L)*HZ
  3   CONTINUE
  2   CONTINUE
  1   CONTINUE
C
         RETURN
	 END
C
C##########################################################################
C THIS CODE IS FOR THE FIELD FROM  2x2x2=8 "CARTESIAN" HARMONICS
C
         SUBROUTINE  BIRK2SHL(X,Y,Z,PS,HX,HY,HZ)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C    The model parameters are provided to this module via common-block /A/.
C  The 16 linear parameters enter in pairs in the amplitudes of the
c       "cartesian" harmonics.
c    The 8 nonlinear parameters are the scales Pi,Ri,Qi,and Si entering the
c  arguments of exponents, sines, and cosines in each of the 8 "Cartesian"
c   harmonics
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
	 IMPLICIT  REAL * 8  (A - H, O - Z)
C
         DIMENSION P(2),R(2),Q(2),S(2)
         DIMENSION A(24)
C
         EQUIVALENCE(P(1),A(17)),(R(1),A(19)),(Q(1),A(21)),(S(1),A(23))
         DATA A/-111.6371348,124.5402702,110.3735178,-122.0095905,
     * 111.9448247,-129.1957743,-110.7586562,126.5649012,-0.7865034384,
     * -0.2483462721,0.8026023894,0.2531397188,10.72890902,0.8483902118,
     * -10.96884315,-0.8583297219,13.85650567,14.90554500,10.21914434,
     * 10.09021632,6.340382460,14.40432686,12.71023437,12.83966657/
C
            CPS=DCOS(PS)
            SPS=DSIN(PS)
            S3PS=4.D0*CPS**2-1.D0   !  THIS IS SIN(3*PS)/SIN(PS)
C
           HX=0.D0
           HY=0.D0
           HZ=0.D0
           L=0
C
           DO 1 M=1,2     !    M=1 IS FOR THE 1ST SUM ("PERP." SYMMETRY)
C                           AND M=2 IS FOR THE SECOND SUM ("PARALL." SYMMETRY)
             DO 2 I=1,2
                  CYPI=DCOS(Y/P(I))
                  CYQI=DCOS(Y/Q(I))
                  SYPI=DSIN(Y/P(I))
                  SYQI=DSIN(Y/Q(I))
C
               DO 3 K=1,2
                   SZRK=DSIN(Z/R(K))
                   CZSK=DCOS(Z/S(K))
                   CZRK=DCOS(Z/R(K))
                   SZSK=DSIN(Z/S(K))
                     SQPR=DSQRT(1.D0/P(I)**2+1.D0/R(K)**2)
                     SQQS=DSQRT(1.D0/Q(I)**2+1.D0/S(K)**2)
                        EPR=DEXP(X*SQPR)
                        EQS=DEXP(X*SQQS)
C
                   DO 4 N=1,2  ! N=1 IS FOR THE FIRST PART OF EACH COEFFICIENT
C                                  AND N=2 IS FOR THE SECOND ONE
C
                    L=L+1
                     IF (M.EQ.1) THEN
                       IF (N.EQ.1) THEN
                         DX=-SQPR*EPR*CYPI*SZRK
                         DY=EPR/P(I)*SYPI*SZRK
                         DZ=-EPR/R(K)*CYPI*CZRK
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ELSE
                         DX=DX*CPS
                         DY=DY*CPS
                         DZ=DZ*CPS
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ENDIF
                     ELSE
                       IF (N.EQ.1) THEN
                         DX=-SPS*SQQS*EQS*CYQI*CZSK
                         DY=SPS*EQS/Q(I)*SYQI*CZSK
                         DZ=SPS*EQS/S(K)*CYQI*SZSK
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                                   ELSE
                         DX=DX*S3PS
                         DY=DY*S3PS
                         DZ=DZ*S3PS
                         HX=HX+A(L)*DX
                         HY=HY+A(L)*DY
                         HZ=HZ+A(L)*DZ
                       ENDIF
                 ENDIF
c
  4   CONTINUE
  3   CONTINUE
  2   CONTINUE
  1   CONTINUE
C
         RETURN
	    END
C*******************************************************************

      SUBROUTINE DIPXYZ(X,Y,Z,BXX,BYX,BZX,BXY,BYY,BZY,BXZ,BYZ,BZZ)
C
C       RETURNS THE FIELD COMPONENTS PRODUCED BY THREE DIPOLES, EACH
C        HAVING M=Me AND ORIENTED PARALLEL TO X,Y, and Z AXIS, RESP.
C
      IMPLICIT REAL*8 (A-H,O-Z)
C
      X2=X**2
      Y2=Y**2
      Z2=Z**2
      R2=X2+Y2+Z2

      XMR5=30574.D0/(R2*R2*DSQRT(R2))
      XMR53=3.D0*XMR5
      BXX=XMR5*(3.D0*X2-R2)
      BYX=XMR53*X*Y
      BZX=XMR53*X*Z
C
      BXY=BYX
      BYY=XMR5*(3.D0*Y2-R2)
      BZY=XMR53*Y*Z
C
      BXZ=BZX
      BYZ=BZY
      BZZ=XMR5*(3.D0*Z2-R2)
C
      RETURN
      END
C
C------------------------------------------------------------
C
         SUBROUTINE INTERCON(X,Y,Z,BX,BY,BZ)
C
C      Calculates the potential interconnection field inside the magnetosphere,
c  corresponding to  DELTA_X = 20Re and DELTA_Y = 10Re (NB#3, p.90, 6/6/1996).
C  The position (X,Y,Z) and field components BX,BY,BZ are given in the rotated
c   coordinate system, in which the Z-axis is always directed along the BzIMF
c   (i.e. rotated by the IMF clock angle Theta)
C   It is also assumed that the IMF Bt=1, so that the components should be
c     (i) multiplied by the actual Bt, and
c     (ii) transformed to standard GSM coords by rotating back around X axis
c              by the angle -Theta.
c
C      Description of parameters:
C
C     X,Y,Z -   GSM POSITION
C      BX,BY,BZ - INTERCONNECTION FIELD COMPONENTS INSIDE THE MAGNETOSPHERE
C        OF A STANDARD SIZE (TO TAKE INTO ACCOUNT EFFECTS OF PRESSURE CHANGES,
C         APPLY THE SCALING TRANSFORMATION)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C     The 9 linear parameters are amplitudes of the "cartesian" harmonics
c     The 6 nonlinear parameters are the scales Pi and Ri entering
c    the arguments of exponents, sines, and cosines in the 9 "Cartesian"
c       harmonics (3+3)
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
        IMPLICIT  REAL * 8  (A - H, O - Z)
C
        DIMENSION A(15),RP(3),RR(3),P(3),R(3)
C
      DATA A/-8.411078731,5932254.951,-9073284.93,-11.68794634,
     * 6027598.824,-9218378.368,-6.508798398,-11824.42793,18015.66212,
     * 7.99754043,13.9669886,90.24475036,16.75728834,1015.645781,
     * 1553.493216/
C
        DATA M/0/
C
        IF (M.NE.0) GOTO 111
        M=1
C
         P(1)=A(10)
         P(2)=A(11)
         P(3)=A(12)
         R(1)=A(13)
         R(2)=A(14)
         R(3)=A(15)
C
C
           DO 11 I=1,3
            RP(I)=1.D0/P(I)
  11        RR(I)=1.D0/R(I)
C
  111   CONTINUE
C
            L=0
C
               BX=0.
               BY=0.
               BZ=0.
C
c        "PERPENDICULAR" KIND OF SYMMETRY ONLY
C
               DO 2 I=1,3
                  CYPI=DCOS(Y*RP(I))
                  SYPI=DSIN(Y*RP(I))
C
                DO 2 K=1,3
                   SZRK=DSIN(Z*RR(K))
                   CZRK=DCOS(Z*RR(K))
                     SQPR=DSQRT(RP(I)**2+RR(K)**2)
                      EPR=DEXP(X*SQPR)
C
                     HX=-SQPR*EPR*CYPI*SZRK
                     HY=RP(I)*EPR*SYPI*SZRK
                     HZ=-RR(K)*EPR*CYPI*CZRK
             L=L+1
c
          BX=BX+A(L)*HX
          BY=BY+A(L)*HY
          BZ=BZ+A(L)*HZ
  2   CONTINUE
C
      RETURN
      END

C~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~