      DOUBLE PRECISION FUNCTION DV2NRM(P, X)
C
C  ***  RETURN THE 2-NORM OF THE P-VECTOR X, TAKING  ***
C  ***  CARE TO AVOID THE MOST LIKELY UNDERFLOWS.    ***
C
      INTEGER P
      DOUBLE PRECISION X(P)
C
      INTEGER I, J
      DOUBLE PRECISION ONE, R, SCALE, SQTETA, T, XI, ZERO
C/+
      DOUBLE PRECISION DSQRT
C/
      DOUBLE PRECISION DR7MDC
      EXTERNAL DR7MDC
C
C/6
C     DATA ONE/1.D+0/, ZERO/0.D+0/
C/7
      PARAMETER (ONE=1.D+0, ZERO=0.D+0)
      SAVE SQTETA
C/
      DATA SQTETA/0.D+0/
C
      IF (P .GT. 0) GO TO 10
         DV2NRM = ZERO
         GO TO 999
 10   DO 20 I = 1, P
         IF (X(I) .NE. ZERO) GO TO 30
 20      CONTINUE
      DV2NRM = ZERO
      GO TO 999
C
 30   SCALE = DABS(X(I))
      IF (I .LT. P) GO TO 40
         DV2NRM = SCALE
         GO TO 999
 40   T = ONE
      IF (SQTETA .EQ. ZERO) SQTETA = DR7MDC(2)
C
C     ***  SQTETA IS (SLIGHTLY LARGER THAN) THE SQUARE ROOT OF THE
C     ***  SMALLEST POSITIVE FLOATING POINT NUMBER ON THE MACHINE.
C     ***  THE TESTS INVOLVING SQTETA ARE DONE TO PREVENT UNDERFLOWS.
C
      J = I + 1
      DO 60 I = J, P
         XI = DABS(X(I))
         IF (XI .GT. SCALE) GO TO 50
              R = XI / SCALE
              IF (R .GT. SQTETA) T = T + R*R
              GO TO 60
 50           R = SCALE / XI
              IF (R .LE. SQTETA) R = ZERO
              T = ONE  +  T * R*R
              SCALE = XI
 60      CONTINUE
C
      DV2NRM = SCALE * DSQRT(T)
 999  RETURN
C  ***  LAST LINE OF DV2NRM FOLLOWS  ***
      END
