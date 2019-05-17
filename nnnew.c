#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 2
#define NUMHID1 2
#define NUMOUT 1

#define rando() ((double)rand()/((double)RAND_MAX+1))

main() {
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumHidden1 = NUMHID1, NumOutput = NUMOUT;
    double* Input;//[NUMPAT+1][NUMIN+1];
    double* Target;//[NUMPAT+1][NUMOUT+1];
    double WeightIH[NUMIN+1][NUMHID+1], WeightHH[NUMHID+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1], Hidden1[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], SumH[NUMPAT+1][NUMHID+1], SumHH[NUMPAT+1][NUMHID+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1], DeltaH1[NUMHID+1], SumDOW1[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHH[NUMHID+1][NUMHID+1],  DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error,eta=0.5, alpha = 0.9, smallwt = 0.5;
    double score,scoresum=0;
  
	FILE* nameOfTheFile = fopen("doc.csv", "r");
	int iter = 0;
	int inputval = (NUMPAT+1)*(NUMIN+1);
	int outputval = (NUMPAT+1)*(NUMOUT+1);
	int val = (NUMPAT+1)*(NUMIN+1)+(NUMPAT+1)*(NUMOUT+1);
	int* arr;
	arr = (int*)malloc(val*sizeof(int));
	Input = (double*)malloc(inputval*sizeof(double));
	Target = (double*)malloc(outputval*sizeof(double));
	int n=0,l,a;
	for(; fscanf(nameOfTheFile, "%d", &iter) && !feof(nameOfTheFile);)
	    arr[n++]= iter;
	fclose(nameOfTheFile);
	int c=0;
	for(l=0;l<(NUMPAT+1);l++)
	{
		for(a=0;a<(NUMIN+1);a++)
		{
			*(Input + l*(NUMIN+1) + a) = arr[c++];
		}
		c+=NUMIN;
	}
	c=0;
	for(l=0;l<(NUMPAT+1);l++)
	{
		c+=3;
		for(a=0;a<(NUMOUT+1);a++)
		{
			*(Target + l*(NUMOUT+1) + a) = arr[c++];
		}
	}
  	
    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    
    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightHH and DeltaWeightHH */
        for( i = 0 ; i <= NumHidden1 ; i++ ) { 
            DeltaWeightHH[i][j] = 0.0 ;
            WeightHH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    
    
    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeisghtHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden1 ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
     
    for( epoch = 0 ; epoch < 100000 ; epoch++) {    /* iterate weight updates */
        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of training patterns */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            np = p + rando() * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit 1 activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += *(Input + p*(NUMIN+1) + i) * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit 2 activations */
                SumHH[p][j] = WeightHH[0][j] ;
                for( i = 1 ; i <= NumHidden1 ; i++ ) {
                    SumHH[p][j] += Hidden[p][i] * WeightHH[i][j] ;
                }
                Hidden1[p][j] = 1.0/(1.0 + exp(-SumHH[p][j])) ;
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden1 ; j++ ) {
                    SumO[p][k] += Hidden1[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                Error += 0.5 * (*(Target + p*(NUMOUT+1) + k) - Output[p][k]) * (*(Target + p*(NUMOUT+1) + k) - Output[p][k]) ;   /* SSE */
/*              Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */
                DeltaO[k] = (*(Target + p*(NUMOUT+1) + k) - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
            }
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden1 ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden1[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ;
                    WeightHO[j][k] += DeltaWeightHO[j][k] ;
                }
            }
            
            for( j = 1 ; j <= NumHidden1 ; j++ ) {    /* 'back-propagate' errors to hidden layer 2 */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH1[j] = SumDOW[j] * Hidden1[p][j] * (1.0 - Hidden1[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightHH */
                DeltaWeightHH[0][j] = eta * DeltaH1[j] + alpha * DeltaWeightHH[0][j] ;
                WeightHH[0][j] += DeltaWeightHH[0][j] ;
                for( i = 1 ; i <= NumHidden1 ; i++ ) { 
                    DeltaWeightHH[i][j] =eta * Hidden[p][i] * DeltaH1[j] + alpha * DeltaWeightHH[i][j];
                    WeightHH[i][j] += DeltaWeightHH[i][j] ;
                }
            }
            
            for( j = 1 ; j <= NumHidden1 ; j++ ) {    /* 'back-propagate' errors to hidden layer 1 */
                SumDOW1[j] = 0.0 ;
                for( k = 1 ; k <= NumHidden ; k++ ) {
                    SumDOW1[j] += WeightHH[j][k] * DeltaH1[k] ;
                }
                DeltaH[j] = SumDOW1[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] =eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) { 
                    DeltaWeightIH[i][j] =eta * (*(Input + p*(NUMIN+1) + i))* DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            
            
        }
        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
    }
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "Input%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "Target%-4d\tOutput%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%f\t", *(Input + p*(NUMIN+1) + i)) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "%f\t%f\t", *(Target + p*(NUMOUT+1) + k), Output[p][k]) ;
            score = *(Target + p*(NUMOUT+1) + k)-Output[p][k];
            if(score<=0)
            score=-1*score;
            scoresum+=score;
        }
    }
    scoresum = scoresum/NumPattern;
    fprintf(stdout, "\n\nScorecard Value -- %f\n\n",scoresum) ;
    fprintf(stdout, "\n\nGoodbye!\n\n") ;
    free(arr);
    free(Input);
    free(Target);
    return 1 ;
}

/*******************************************************************************/
