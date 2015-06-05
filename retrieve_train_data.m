%
% Load the training data for the approximate speed source identification
% problem. The machine learning components of this software package are
% from the Scattering Transform package by Burna and Mallat; see original
% license below.
%
%
%
%� CMAP, Ecole Polytechnique 
%contributors: Joan Bruna and St�phane Mallat (2010)
%bruna@cmap.polytechnique.fr
%
%This software is a computer program whose purpose is to implement
%an image classification architecture based on the Scattering Transform.
%
%This software is governed by the CeCILL  license under French law and
%abiding by the rules of distribution of free software.  You can  use, 
%modify and/ or redistribute the software under the terms of the CeCILL
%license as circulated by CEA, CNRS and INRIA at the following URL
%"http://www.cecill.info". 
%
%As a counterpart to the access to the source code and  rights to copy,
%modify and redistribute granted by the license, users are provided only
%with a limited warranty  and the software's author,  the holder of the
%economic rights,  and the successive licensors  have only  limited
%liability. 
%
%In this respect, the user's attention is drawn to the risks associated
%with loading,  using,  modifying and/or developing or reproducing the
%software by the user in light of its specific status of free software,
%that may mean  that it is complicated to manipulate,  and  that  also
%therefore means  that it is reserved for developers  and  experienced
%professionals having in-depth computer knowledge. Users are therefore
%encouraged to load and test the software's suitability as regards their
%requirements in conditions enabling the security of their systems and/or 
%data to be ensured and,  more generally, to use and operate it in the 
%same conditions as regards security. 
%
%The fact that you are presently reading this means that you have had
%knowledge of the CeCILL license and that you accept its terms.
%
%
%

function [train, test] = retrieve_train_data(Ltrain, Ltest, maxclasses)

if nargin < 3
D=2;
maxclasses=[0:D-1];
else
D=length(maxclasses);
end

load(fullfile('approx_speed_training_data.mat'));

train_labels=approx_speed_train(:,1);
test_labels=approx_speed_test(:,1);

dim = sqrt(size(approx_speed_train,2) - 1);
for d=1:D
	sli = find(train_labels==d-1);
	for s=1:min(Ltrain,length(sli))
			train{d}{s} = reshape(approx_speed_train(sli(s),2:end),dim,dim);
	end
	sli = find(test_labels==d-1);
	for s=1:min(Ltest,length(sli))
			test{d}{s} = reshape(approx_speed_test(sli(s),2:end),dim,dim);
	end
end
size(train{1}),size(test{1}),size(train{2}),size(test{2})

clear usps_test;
clear usps_train;


