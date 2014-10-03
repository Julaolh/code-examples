#include "ScoresOptimizer.h"
#include "cRandom.h"
#include <deque>
using std::deque;

CScoresOptimizer::CScoresOptimizer(cRandom* pRnd) : m_bMakePiCycle(false), m_nPiCycleAttractorType(0)
{
	m_pRnd = pRnd;
}


CScoresOptimizer::~CScoresOptimizer()
{
}


void CScoresOptimizer::OptimizeScores( SDataStore* pData )
{
	m_nPatterns = pData->vPatFactScores.size();
	m_nFactors = pData->vFactPatScores.size();
	m_nNeurons = pData->vNeurProbSpecific.size();
	if( m_nFactors == 0 )
		return;

	// change prorbability of neuron in factor from >0.99 on 0.99.
	for(std::vector<vector<pair<int,double> > >::iterator itNeurProb = pData->vFactNeurProb.begin(); itNeurProb != pData->vFactNeurProb.end(); ++itNeurProb)
	{
		for(std::vector<pair<int,double> >::iterator it = itNeurProb->begin(); it != itNeurProb->end(); ++it)
		{		
			if( it->second > 0.99999 )
				it->second = 0.99999;
		}
	}
	for(std::vector<vector<pair<int,double> > >::iterator itFactProb = pData->vNeurFactProb.begin(); itFactProb != pData->vNeurFactProb.end(); ++itFactProb)
	{
		for(std::vector<pair<int,double> >::iterator it = itFactProb->begin(); it != itFactProb->end(); ++it)
		{		
			if( it->second > 0.99999 )
				it->second = 0.99999;
		}
	}
	//double fLoading, fScore, fFactor;
	//cout << ComputeLikelihood(pData, fLoading, fScore, fFactor) / (m_nNeurons * m_nPatterns * log(2.0) ) << "\n";
	m_pData = pData;
	m_vFactProb = pData->vFactProb;
	m_vTestFactorIndices.clear();
	vector<pair<int, double> > vSortedFactorIndices;
	for(int l = 0; l != m_pData->vFactProb.size(); ++l)
		vSortedFactorIndices.push_back( make_pair(l, m_pData->vFactProb[l]) );
	std::sort(vSortedFactorIndices.begin(), vSortedFactorIndices.end(), UseStl::less2nd<pair<int, double> >());
	for(std::vector<pair<int, double> >::iterator it = vSortedFactorIndices.begin(); it != vSortedFactorIndices.end(); ++it)
		m_vTestFactorIndices.push_back( it->first );
	m_pRnd->RandomShuffle(m_vTestFactorIndices); // change the order of traversal of factors - randomness

	if( m_bMakePiCycle )
	{
		MakePiCycles(pData->vPatFactScores);
	}
	else
	{
		pData->vPatFactScores.assign(m_nPatterns, vector<int>()); // reset scores - randomness
		for(int k = 0; k != m_nPatterns; ++k)
		{
			vector<int>& vScores = pData->vPatFactScores[k];
//			cout << vScores.size() << "\t\t";
			MaximizeLikelihood(k, vScores);
		}
		ComputeFactProb( pData->vPatFactScores );
	}

	pData->vFactPatScores.assign(m_nFactors, vector<int>());
	for(row_iterator itRow = pData->vPatFactScores.begin(); itRow != pData->vPatFactScores.end(); ++itRow)
	{
		int nPattern = itRow - pData->vPatFactScores.begin();
		for(std::vector<int>::iterator it = itRow->begin(); it != itRow->end(); ++it)
			pData->vFactPatScores[*it].push_back( nPattern );
	}
	pData->vFactProb = m_vFactProb;
}


double CScoresOptimizer::ComputeLikelihood(const SDataStore* pData, double& fLoadingLikelihood, double& fScoreLikelihood, double& fFactorLikelihood)
{
	double fLikelihood = 0;
	fLoadingLikelihood = fScoreLikelihood = fFactorLikelihood = 0;
	int nFactors = pData->vFactPatScores.size();
	int nPatterns = pData->vPatFactScores.size();
	for(int k = 0; k != nPatterns; ++k)
	{
		vector<int> vDenseScores(nFactors, 0);
		for(std::vector<int>::const_iterator it = pData->vPatFactScores[k].begin(), itEnd = pData->vPatFactScores[k].end(); it != itEnd; ++it)
			vDenseScores[*it] = 1;
		pair<double,double> paLikelihood = ComputeLikelihood(k, pData, vDenseScores, pData->vFactProb);
		fLikelihood += paLikelihood.first + paLikelihood.second;
		fLoadingLikelihood += paLikelihood.first;
		fScoreLikelihood += paLikelihood.second;
	}
	int nNeurons = pData->vNeurProbSpecific.size();
	vector<int> vFactNeurCount;
	for(int l = 0; l != nFactors; ++l)
		vFactNeurCount.push_back( pData->vFactNeurProb[l].size() );
	for(int l = 0; l != nFactors; ++l)
	{
		double fProb = (double)vFactNeurCount[l] / nNeurons;
		const vector<pair<int,double> >& vNeurProb = pData->vFactNeurProb[l];
		std::vector<pair<int,double> >::const_iterator itNeurProb = vNeurProb.begin();
		for(int i = 0; i != nNeurons; ++i)
		{
			while( itNeurProb != vNeurProb.end() && itNeurProb->first < i )
				++itNeurProb;
			if( itNeurProb != vNeurProb.end() && itNeurProb->first == i )
			{
				fFactorLikelihood += (fProb == 0) ? -123456789 : log( fProb );
			}
			else
			{
				fFactorLikelihood += (fProb > 0.5) ? 0 : log( 1 - fProb );
			}
		}
	}
	fLikelihood += fFactorLikelihood;
	return fLikelihood;
}


double CScoresOptimizer::ComputeLikelihood0(const SDataStore* pData)
{
	int nPatterns = pData->vPatFactScores.size();
	int nNeurons = pData->vNeurProbSpecific.size();
	// Calculation of the likelihood without factors
	double fLikelihood = 0;
	for(int k = 0; k != nPatterns; ++k)
	{
		const vector<int>& vPattern = pData->vPatterns[k];
		std::vector<int>::const_iterator itPatNeur = vPattern.begin();
		for(int i = 0; i != nNeurons; ++i)
		{
			double fNeurProb = (double)pData->vNeurActPat[i].size() / nPatterns;
			while( itPatNeur != vPattern.end() && *itPatNeur < i )
				++itPatNeur;
			if( itPatNeur != vPattern.end() && *itPatNeur == i )
			{
				fLikelihood += fNeurProb == 0 ? -123456789 : log( fNeurProb );
				++itPatNeur;
			}
			else
				fLikelihood += (1 - fNeurProb) == 0 ? -123456789 : log( 1 - fNeurProb );
		}
	}
	return fLikelihood;
}


pair<double, double> CScoresOptimizer::ComputeIfIq(int l, const vector<int>& vPattern, const vector<int>& vDenseScores, const vector<double>& vFactProb, const SDataStore* pData)
{
	double If = 0, Iq = 0;
	const vector<pair<int,double> >& vNeurProb = pData->vFactNeurProb[l];
	
	std::vector<int>::const_iterator itPatNeur = vPattern.begin();
	for(std::vector<pair<int,double> >::const_iterator itFactNeur = vNeurProb.begin(); itFactNeur != vNeurProb.end(); ++itFactNeur)
	{
		int i = itFactNeur->first;
		double fProduct = 1 - max(pData->vNeurProbSpecific[i], 0.01); // q_j > 0 && p_{ij} > 0 => qi < 1; fi > 0; fi < 1 iff p_{ij} < 1
		const vector<pair<int, double> >& vFactProb = pData->vNeurFactProb[i];
		
		int n1 = 0;
		for(std::vector<pair<int, double> >::const_iterator it = vFactProb.begin(); it != vFactProb.end(); ++it)
			if( it->first != l && vDenseScores[it->first] == 1 && ++n1 )
				fProduct *= 1 - it->second;
		double fi = 1 - fProduct * (1 - itFactNeur->second), qi = 1 - fProduct;
		while( itPatNeur != vPattern.end() && *itPatNeur < itFactNeur->first )
			++itPatNeur;
		if( itPatNeur != vPattern.end() && *itPatNeur == itFactNeur->first )
		{
			If += fi == 0 ? -123456789 : log( fi );
			Iq += qi == 0 ? -123456789 : log( qi );
			if(  fi == 0 || qi == 0 )
				int a = 0;
		}
		else
		{
			If += (1 - fi) == 0 ? -123456789 : log( 1 - fi );
			Iq += (1 - qi) == 0 ? -123456789 : log( 1 - qi );
			if(  fi == 1 || qi == 1 )
				int a = 0;
		}
	}
	if( vFactProb[l] != 0 )
		If += log( vFactProb[l] );
	Iq += (1 - vFactProb[l]) == 0 ? -123456789 : log( 1 - vFactProb[l] );
	if( vFactProb[l] == 1 || !(If > -1000000 && If < 1000000) || !(Iq > -1000000 && Iq < 1000000) )
		int a = 0;
	
	return make_pair(If, Iq);
}


pair<double,double> CScoresOptimizer::ComputeLikelihood(int k, const SDataStore* pData, const vector<int>& vDenseScores, const vector<double>& vFactProb)
{
	const vector<int>& vPattern = pData->vPatterns[k];
	double fLoadingLikelihood = 0;
	int nNeurons = pData->vNeurProbSpecific.size();
	std::vector<int>::const_iterator itPatNeur = vPattern.begin();
	for(int i = 0; i != nNeurons; ++i)
	{
		const vector<pair<int, double> >& vFactProb = pData->vNeurFactProb[i];
		double fProduct = 1 - max(pData->vNeurProbSpecific[i], 0.00001);
		if( pData->vNeurProbSpecific[i] < 0 || pData->vNeurProbSpecific[i] > 1 )
			int a = 0;
		for(std::vector<pair<int, double> >::const_iterator it = vFactProb.begin(); it != vFactProb.end(); ++it)
			if( vDenseScores[it->first] != 0 )
				fProduct *= 1 - it->second;
		if( itPatNeur != vPattern.end() && *itPatNeur == i )
		{
			if( (1 - fProduct) != 0 )
				fLoadingLikelihood += log( 1 - fProduct );
			else
				fLoadingLikelihood =  -123456789;
			++itPatNeur;
		}
		else
		{
			if( fProduct != 0 )
				fLoadingLikelihood += log( fProduct );
			else
				fLoadingLikelihood = -123456789;
		}
	}
	double fScoreLikelihood = 0;
	for(int l = 0; l != vDenseScores.size(); ++l)
	{
		if( vDenseScores[l] != 0 )
			fScoreLikelihood += vFactProb[l] == 0 ? -123456789 : log( vFactProb[l] );
		else
			fScoreLikelihood += (1 - vFactProb[l]) == 0 ? -123456789 : log( 1 - vFactProb[l] );
	}
	return make_pair(fLoadingLikelihood, fScoreLikelihood);
}


int CScoresOptimizer::MakePiCycles( vector<vector<int> >& vPatFactScores )
{
	vector<vector<int> > vPrevPatFactScores = vPatFactScores, vPrevPrevPatFactScores;
	int nCycle, nDifference;
	int nStep = 0;
	do 
	{
		std::swap(vPrevPrevPatFactScores, vPrevPatFactScores);
		vPrevPatFactScores= vPatFactScores;

		for(int k = 0; k != m_nPatterns; ++k)
		{
			vector<int>& vScores = vPatFactScores[k];
			MaximizeLikelihood(k, vScores);
		}

		ComputeFactProb( vPatFactScores );

		if( !m_bMakePiCycle )
			break;

		int nDiff1 = 0;
		for(row_iterator itRow1 = vPatFactScores.begin(), itRow2 = vPrevPatFactScores.begin(); itRow1 != vPatFactScores.end(); ++itRow1, ++itRow2)
		{
			int OvInt = mylib::OverlapSimple(*itRow1, *itRow2);
			nDiff1 += max(itRow1->size(), itRow2->size()) - OvInt;
		}
		int nDiff2 = 0;
		for(row_iterator itRow1 = vPatFactScores.begin(), itRow2 = vPrevPrevPatFactScores.begin(); itRow1 != vPatFactScores.end(); ++itRow1, ++itRow2)
		{
			int OvInt = mylib::OverlapSimple(*itRow1, *itRow2);
			nDiff2 += max(itRow1->size(), itRow2->size()) - OvInt;
		}
		nDifference = min(nDiff1, nDiff2);
		nCycle = nDiff1 < nDiff2 ? 1 : 2;

		double fLikelihood = 0;
		double fScoreLikelihood = 0;
		for(int k = 0; k != m_nPatterns; ++k)
		{
			pair<double,double> paLikelihood = ComputeLikelihood(k, m_pData, vPatFactScores[k], m_vFactProb);
			fLikelihood += paLikelihood.first + paLikelihood.second;
			fScoreLikelihood += paLikelihood.second;
		}
		cout << "Step = " << ++nStep << "\t" << "Total likelihood = " << fLikelihood / (m_nPatterns * m_nNeurons * log(2.0)) << "\t" << "Difference = " << nDifference << "\t" << "ScoreLikelihood = " << fScoreLikelihood / (m_nPatterns * m_nNeurons * log(2.0)) << "\n";

		m_vCycles.resize(m_vCycles.size()+1);
		t_pi_cycle& tCycle = m_vCycles.back();
		tCycle.fLikelihood = fLikelihood;
		tCycle.fScoreLikelihood = fScoreLikelihood;
		tCycle.nDifference = nDifference;

	} while( nDifference > 0 );
	m_nPiCycleAttractorType = nCycle;
	return nStep;
}




// asynchronous maximization algorithm
int CScoresOptimizer::MaximizeLikelihood(int k, vector<int>& vScores)
{
	const vector<int>& vPattern = m_pData->vPatterns[k];
	if( vPattern.empty() )
	{
		vScores.clear();
		return 0;
	}
	if( m_vTestFactorIndices.empty() )
		return 0;
	int nFactors = m_vTestFactorIndices.size();

	vector<int> vDenseScores(nFactors, 0);
	for(std::vector<int>::iterator it = vScores.begin(); it != vScores.end(); ++it)
		vDenseScores[*it] = 1;



	int nStep = 0;
	deque<int> dScoreDifference(nFactors, 1);
	std::vector<int>::iterator itTestFactor = m_vTestFactorIndices.begin();
	do 
	{
		int l = *itTestFactor;
		pair<double, double> IfIq = ComputeIfIq(l, vPattern, vDenseScores, m_vFactProb, m_pData);
		int& nScore = vDenseScores[l];
		int nPrevScore = nScore;
		nScore = ( IfIq.first > IfIq.second + 0.00000001 ) ? 1 : 0;
		dScoreDifference.pop_front();
		dScoreDifference.push_back( (nScore == nPrevScore) ? 0 : 1 );
		if( ++itTestFactor == m_vTestFactorIndices.end() )
		{
			m_pRnd->RandomShuffle(m_vTestFactorIndices);
			itTestFactor = m_vTestFactorIndices.begin();
		}
		++nStep;
		//pair<double,double> paLikelihood = ComputeLikelihood(k, m_pData, vDenseScores, m_vFactProb);
		//cout << k << "\t" << nStep << "\t" << paLikelihood.first + paLikelihood.second << "\n";
	} while( std::accumulate(dScoreDifference.begin(), dScoreDifference.end(), 0) != 0 );

	

	vScores.clear();
	for(std::vector<int>::iterator it = vDenseScores.begin(); it != vDenseScores.end(); ++it)
		if( *it != 0 )
			vScores.push_back( it - vDenseScores.begin() );

	return nStep;
}


void CScoresOptimizer::ComputeFactProb( vector<vector<int> >& vPatFactScores )
{
	m_vFactProb.assign( m_nFactors, 0 );
	for(row_iterator itRow = vPatFactScores.begin(); itRow != vPatFactScores.end(); ++itRow)
	{
		int nPattern = itRow - vPatFactScores.begin();
		for(std::vector<int>::iterator it = itRow->begin(); it != itRow->end(); ++it)
			m_vFactProb[ *it ]++;
	}
	if( m_nPatterns )
		std::transform(m_vFactProb.begin(), m_vFactProb.end(), m_vFactProb.begin(), std::bind2nd(std::multiplies<double>(), 1.0 / m_nPatterns) );
}
