/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

#define DBL_MAX (1e10)
#define EPS (0.00001)
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // set number of particles
  num_particles = 50;
  
  // Noisy GPS data for particle generation
  // mu + N(0,std) -> N(mu,std)
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x,std[0]); 
  std::normal_distribution<double> dist_y(y,std[1]);
  std::normal_distribution<double> dist_theta(theta,std[2]);
  for (int id = 0; id < num_particles; id++) {
    particles.emplace_back(Particle {id, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0});
    weights.emplace_back(1.0);
  }
  
  is_initialized = true;

}
void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  // define random number generator engine
  std::default_random_engine gen;
  // Additive White Gaussian Noise (AWGN)
  std::normal_distribution<double> awgn_x(0,std_pos[0]);
  std::normal_distribution<double> awgn_y(0,std_pos[1]);
  std::normal_distribution<double> awgn_yaw(0,std_pos[2]);
  
  for (int i=0; i < num_particles; ++i) {
    double theta = particles[i].theta;
    // for very small yaw rate we can ignore it
    if (fabs(yaw_rate) < EPS) {
      particles[i].x += velocity * cos(theta) * delta_t + awgn_x(gen);
      particles[i].y += velocity * sin(theta) * delta_t + awgn_y(gen);

      particles[i].theta += awgn_yaw(gen);
    } else {
      particles[i].x += velocity/yaw_rate * ( sin(theta + yaw_rate*delta_t) -
                                                sin(theta)) + awgn_x(gen);
      particles[i].y += velocity/yaw_rate * (-cos(theta + yaw_rate*delta_t) +
                                                cos(theta)) + awgn_y(gen);
      
      particles[i].theta += yaw_rate*delta_t + awgn_yaw(gen);
    }
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  for(size_t i = 0; i < observations.size(); ++i) {
    int assigned_predicted_landmark_{0};
    double min_distance = dist(observations[i].x, observations[i].y,
                               predicted[0].x, predicted[0].y);
    
    for(size_t j = 1; j < predicted.size(); ++j) {
      double distance = dist(observations[i].x, observations[i].y,
                             predicted[j].x, predicted[j].y);
      
      if (distance < min_distance) {
        min_distance = distance;
        assigned_predicted_landmark_ = j;
      }
    }
    observations[i].id = assigned_predicted_landmark_;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  double sig2_x = std_landmark[0] * std_landmark[0];
  double sig2_y = std_landmark[1] * std_landmark[1];
  double sig_xy= std_landmark[1] * std_landmark[1];
  double normalizer = 2.0 * M_PI * sig_xy;

  // compute weight for each particle
  for(int i = 0; i < num_particles; ++i) {
    
    double x_p = particles[i].x;
    double y_p = particles[i].y;
    double yaw_p = particles[i].theta;
    
    // find land marks that are located in sensor range
    vector<LandmarkObs> predictions;
    inSensorRange(x_p, y_p, sensor_range, map_landmarks, predictions);

    // when there is no land mark in range of sensor set weight to zero
    if(predictions.size() == 0) {
      particles[i].weight = 0;
      weights[i] = 0;
    }
    else {
      // Transformation sensed landmark position to global coordinates
      vector<LandmarkObs> mapped_obs;
      for(size_t j = 0; j < observations.size(); ++j) {
        
        auto obs = observations[j];
        [&x_p, &y_p, &yaw_p](LandmarkObs &obs) {
            double x = x_p + cos(yaw_p) * obs.x - sin(yaw_p) * obs.y;
            double y = y_p + sin(yaw_p) * obs.x + cos(yaw_p) * obs.y;
            obs.x = x;
            obs.y = y;}(obs);
        mapped_obs.emplace_back(obs);
      }
      // find most likely match between observation and sensed land mark
      dataAssociation(predictions, mapped_obs);

      // total probablity through production of multi-variate Gaussians
      double weight = 1.0;
      for(size_t k = 0; k < mapped_obs.size(); ++k) {
        auto obs = mapped_obs[k];
        auto predicted = predictions[obs.id];

        double dx = obs.x - predicted.x;
        double dy = obs.y - predicted.y;
        weight *= exp(-(dx * dx / (2 * sig2_x) + dy * dy / (2 * sig2_y))) / normalizer;
      }
      particles[i].weight = weight;
      weights[i] = weight;
    }
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> resampled_particles;
  std::default_random_engine gen;
  std::discrete_distribution<int> random_index(weights.begin(), weights.end());

  for (int i=0; i<num_particles; ++i)
  resampled_particles.push_back( particles[random_index(gen)] );
  
  particles = std::move(resampled_particles);
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}