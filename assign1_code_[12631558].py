#!/usr/bin/env python

import rospy
import math
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
# import cv2 as cv # OpenCV2
import random
import numpy as np
import copy
import tf2_ros
from threading import Lock
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def random_uniform(a, b):
    # Returns a random number with uniform distribution between "a" and "b"
    if b < a:
        rospy.logerror("The first argument must be less than the second argument when using the random_uniform method")
    return random.uniform(a, b)

def random_normal(stddev):
    # Returns a random number with normal distribution, 0 mean and a standard deviation of "stddev"
    return np.random.normal(0.0, stddev)

class ParticleFilter:
    def __init__(self):

        # Parameters
        # Default values will be used if no ROS parameter exists (see launch file)
        self.num_particles_ = rospy.get_param("~num_particles", 500) # Number of particles
        self.num_motion_updates_ = rospy.get_param("~num_motion_updates", 5) # Number of motion updates before a sensor update
        self.num_scan_rays_ = rospy.get_param("~num_scan_rays", 6) # (Approximate) number of scan rays to evaluate
        self.num_sensing_updates_ = rospy.get_param("~num_sensing_updates", 5) # Number of sensing updates before resampling
        self.motion_distance_noise_stddev_ = rospy.get_param("~motion_distance_noise_stddev", 0.01) # Standard deviation of distance noise for motion update
        self.motion_rotation_noise_stddev_ = rospy.get_param("~motion_rotation_noise_stddev", math.pi / 60.) # Standard deviation of rotation noise for motion update
        self.sensing_noise_stddev_ = rospy.get_param("~sensing_noise_stddev", 0.5) # Standard deviation of sensing noise

        # Get the map via a ROS service call
        rospy.loginfo("Waiting for static_map service...")
        rospy.wait_for_service('static_map')
        get_map_service = rospy.ServiceProxy('static_map', GetMap)
        try:
            resp = get_map_service()
            self.map_ = resp.map
        except rospy.ServiceException as exc:
            rospy.logerror('Service did not process request: ' + str(exc))
            rospy.signal_shutdown()
        rospy.loginfo("Map received")

        # Convert occupancy grid into a numpy array
        self.map_image_ = np.reshape(self.map_.data, (self.map_.info.height, self.map_.info.width)).astype(np.int32)
        print(self.map_image_)

        # Limits of map in meters
        self.map_x_min_ = self.map_.info.origin.position.x
        self.map_x_max_ = self.map_.info.width * self.map_.info.resolution + self.map_.info.origin.position.x
        self.map_y_min_ = self.map_.info.origin.position.y
        self.map_y_max_ = self.map_.info.height * self.map_.info.resolution + self.map_.info.origin.position.y

        # Variables
        self.particles_ = [] # Vector that holds the particles
        self.prev_odom_msg_ = None # Stores the previous odometry message to determine distance travelled
        self.estimated_pose_ = Pose() 

        # Counters
        self.motion_update_count_ = 0 # Number of motion updates since last sensor update
        self.sensing_update_count_ = 0 # Number of sensing updates since resampling

        # Create a thread lock to avoid having different callbacks changing the particles_
        self.lock_ = Lock()
        self.lock_.acquire()

        # ROS Subscribers
        self.odom_sub_ = rospy.Subscriber('odom', Odometry, self.odom_callback, queue_size=1) # Subscribes to wheel odometry
        self.scan_sub_ = rospy.Subscriber('base_scan', LaserScan, self.scan_callback, queue_size=1) # Subscribes to laser scan

        # ROS Publishers
        self.particles_pub_ = rospy.Publisher('particles', PoseArray, queue_size=1)
        self.particles_seq_ = 0
        self.particles_pub_timer_ = rospy.Timer(rospy.Duration(0.1), self.publish_particles)

        self.estimated_pose_pub_ = rospy.Publisher('estimated_pose', PoseStamped, queue_size=1)
        self.estimated_pose_seq_ = 0
        self.estimated_pose_pub_timer_ = rospy.Timer(rospy.Duration(0.1), self.publish_estimated_pose)

        self.transform_broadcaster_ = tf2_ros.TransformBroadcaster()
        self.transform_seq_ = 0

        # Marker used for debug visualisation
        self.marker_ = Marker()
        self.marker_.header.frame_id = "map"
        self.marker_.ns = "ns"
        self.marker_.id = 0
        self.marker_.type = Marker.LINE_LIST
        self.marker_.action = Marker.ADD
        self.marker_.pose.position.x = 0.0
        self.marker_.pose.position.y = 0.0
        self.marker_.pose.position.z = 0.0
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0
        self.marker_.scale.x = 0.01
        self.marker_.scale.y = 0.1
        self.marker_.scale.z = 0.1
        self.marker_.color.a = 1.0
        self.marker_.color.r = 0.0
        self.marker_.color.g = 0.7
        self.marker_.color.b = 0.2
        self.marker_pub_ = rospy.Publisher('marker', Marker, queue_size=1)

        # Initialise particles
        self.initialise_particles()

        self.lock_.release()

    def initialise_particles(self):
        # "num_particles_" is the number of particles you will create

        # Clear the particles array
        self.particles_ = []

        # You want to initialise the particles in the "particles_" array
        # "random_uniform(a, b)" will give you a random value with uniform distribution between "a" and "b"
        # "self.map_x_min_", "self.map_x_max_", "self.map_y_min_", and "self.map_y_max_" give you the limits of the map
        # Orientation (theta) should be between 0 and 2*Pi
        # You probably need to use a "." in your numbers (e.g. "1.0") when calculating the weights

        ####################
        ## YOUR CODE HERE ##
        ####################
        for i in range(self.num_particles_):
            x = random_uniform(self.map_x_min_, self.map_x_max_)
            y = random_uniform(self.map_y_min_, self.map_y_max_)
            localTheta = random_uniform(0, 2*math.pi)
        
            self.particles_.append(Particle(x, y, localTheta, 1.0))
        



        # Particles may be initialised in occupied space but the map has thin walls so it should be OK
        # TODO inflate the occupancy grid and check that particles are not in occupied space

        # Don't use the estimated the pose just after initialisation
        self.estimated_pose_valid_ = False;

        # Induce a sensing update
        self.motion_update_count_ = self.num_motion_updates_;
        
    def normalise_weights(self):
        # Normalise the weights of the particles in "particles_"

        ####################
        ## YOUR CODE HERE ##
        ####################

        totalWeight = 0
        for index in range(len(self.particles_)):
            totalWeight += self.particles_[index].weight

        for index in range(len(self.particles_)):
            self.particles_[index].weight = self.particles_[index].weight/totalWeight


    def hit_scan(self, start_x, start_y, theta, max_range, draw=False):
        # Find the nearest obstacle from position start_x, start_y (in metres) in direction theta

        # Start point in occupancy grid coordinates
        start_point = [int(round((start_x - self.map_.info.origin.position.x) / self.map_.info.resolution)),
                             int(round((start_y - self.map_.info.origin.position.y) / self.map_.info.resolution))];

        # End point in real coordinates
        end_x = start_x + math.cos(theta) * max_range;
        end_y = start_y + math.sin(theta) * max_range;

        # End point in occupancy grid coordinates
        end_point = [int(round((end_x - self.map_.info.origin.position.x) / self.map_.info.resolution)),
                           int(round((end_y - self.map_.info.origin.position.y) / self.map_.info.resolution))];

        # Find the first "hit" along scan
        # (unoccupied is value 0, occupied is value 100)
        hit = find_hit(self.map_image_, start_point, end_point, 50)

        # Convert hit back to world coordinates
        hit_x = hit[0] * self.map_.info.resolution + self.map_.info.origin.position.x
        hit_y = hit[1] * self.map_.info.resolution + self.map_.info.origin.position.y

        # Add a debug visualisation marker
        if draw:
            point = Point(start_x, start_y, 0.)
            self.marker_.points.append(point)
            point = Point(hit_x, hit_y, 0.)
            self.marker_.points.append(point)

        # Get distance to hit
        return math.sqrt(math.pow(start_x - hit_x, 2) + math.pow(start_y - hit_y, 2))


    def estimate_pose(self):
        # Position of the estimated pose
        estimated_pose_x = 0.0
        estimated_pose_y = 0.0
        estimated_pose_theta = 0.0

        # Choose a method to estimate the pose from the particles in the "particles_" vector
        # Put the values into "estimated_pose_x", "estimated_pose_y", and "estimated_pose_theta"
        # If you just use the pose of the particle with the highest weight the maximum mark you can get for this part is 0.5


        ####################
        ## YOUR CODE HERE ##
        ####################
        counter = 0
        poseX = 0.0
        poseY = 0.0
        poseTheta = 0.0
        
        for index in range(len(self.particles_)):
             poseX += self.particles_[index].x
             poseY += self.particles_[index].y
             poseTheta += self.particles_[index].theta
             counter += 1
    
        estimated_pose_x = poseX/counter
        estimated_pose_y = poseY/counter
        estimated_pose_theta = poseTheta/counter


        # Set the estimated pose message
        self.estimated_pose_.position.x = estimated_pose_x;
        self.estimated_pose_.position.y = estimated_pose_y;

        self.estimated_pose_.orientation.w = math.cos(estimated_pose_theta / 2.);
        self.estimated_pose_.orientation.z = math.sin(estimated_pose_theta / 2.);

        self.estimated_pose_valid_ = True;


    def resample_particles(self):
        # Resample the particles
        # Weights are expected to be normalised

        # Move old particles
        old_particles = copy.deepcopy(self.particles_)
        self.particles_ = []

        # Iterator for old_particles
        old_particles_i = 0

        while len(self.particles_) < self.num_particles_:
            value = random_uniform(0.0, 1.0)
            sum = 0.0

            # Loop until a particle is found
            particle_found = False
            while not particle_found:

                # If the random value is between the sum and the sum + the weight of the particle
                if value > sum and value < (sum + old_particles[old_particles_i].weight):

                    # Add the particle to the "particles_" vector
                    self.particles_.append(copy.deepcopy(old_particles[old_particles_i]))

                    # Add jitter to the particle
                    self.particles_[-1].x = self.particles_[-1].x + random_normal(0.02)
                    self.particles_[-1].y = self.particles_[-1].y + random_normal(0.02)
                    self.particles_[-1].theta = wrap_angle(self.particles_[-1].theta + random_normal(math.pi / 30.));

                    # The particle may be out of the map, but that will be fixed by the motion update
                    
                    # Break out of the loop
                    particle_found = True

                # Add particle weight to sum and increment the iterator
                sum = sum + old_particles[old_particles_i].weight;
                old_particles_i = old_particles_i + 1;

                # If the iterator is past the vector, loop back to the beginning
                if old_particles_i >= len(old_particles):
                    old_particles_i = 0

        # Normalise the new particles
        self.normalise_weights();

        # Don't use the estimated pose just after resampling
        self.estimated_pose_valid_ = False;

        # Induce a sensing update
        self.motion_update_count_ = self.num_motion_updates_;        


    def publish_particles(self, event):
        self.lock_.acquire()
        pose_array = PoseArray()

        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "map"
        pose_array.header.seq = self.particles_seq_
        self.particles_seq_ = self.particles_seq_ + 1

        for p in self.particles_:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.orientation.w = math.cos(p.theta / 2.)
            pose.orientation.z = math.sin(p.theta / 2.)

            pose_array.poses.append(pose);

        self.particles_pub_.publish(pose_array)
        self.lock_.release()

    def publish_estimated_pose(self, event):        
        if not self.estimated_pose_valid_:
            return

        self.lock_.acquire()

        # Publish the estimated pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.seq = self.estimated_pose_seq_
        self.estimated_pose_seq_ = self.estimated_pose_seq_ + 1

        pose_stamped.pose = self.estimated_pose_

        self.estimated_pose_pub_.publish(pose_stamped)

        # Broadcast "map" to "base_footprint" transform
        transform = TransformStamped()

        transform.header.frame_id = "map"
        transform.header.stamp = rospy.Time.now()
        transform.header.seq = self.transform_seq_
        self.transform_seq_ = self.transform_seq_ + 1

        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = self.estimated_pose_.position.x
        transform.transform.translation.y = self.estimated_pose_.position.y

        transform.transform.rotation.w = self.estimated_pose_.orientation.w
        transform.transform.rotation.z = self.estimated_pose_.orientation.z

        self.transform_broadcaster_.sendTransform(transform)
        self.lock_.release()

    def odom_callback(self, odom_msg):
        # Receive an odometry message

        # Skip the first call since we are looking for movements
        if not self.prev_odom_msg_:
            self.prev_odom_msg_ = odom_msg
            return

        # Distance moved since the previous odometry message
        global_delta_x = odom_msg.pose.pose.position.x - self.prev_odom_msg_.pose.pose.position.x;
        global_delta_y = odom_msg.pose.pose.position.y - self.prev_odom_msg_.pose.pose.position.y;

        distance = math.sqrt(math.pow(global_delta_x, 2.) + math.pow(global_delta_y, 2.));

        # Previous robot orientation
        prev_theta = 2 * math.acos(self.prev_odom_msg_.pose.pose.orientation.w)

        if self.prev_odom_msg_.pose.pose.orientation.z < 0.:
            prev_theta = -prev_theta

        # Figure out if the direction is backward
        if (prev_theta < 0. and global_delta_y > 0.) or (prev_theta > 0. and global_delta_y < 0.):
            distance = -distance

        # Current orientation
        theta = 2 * math.acos(odom_msg.pose.pose.orientation.w)

        if odom_msg.pose.pose.orientation.z < 0.:
            theta = -theta

        # Rotation since the previous odometry message
        rotation = theta - prev_theta

        # Return if the robot hasn't moved
        if distance == 0 and rotation == 0:
            return

        self.lock_.acquire()

        # Motion update: add "distance" and "rotation" to each particle
        # You also need to add noise, which should be different for each particle
        # Use "random_normal()" with "self.motion_distance_noise_stddev_" and "self.motion_rotation_noise_stddev_" to get random values
        # You will probably need "math.cos()" and "math.sin()", and you should wrap theta with "wrap_angle()" too


        ####################
        ## YOUR CODE HERE ##
        ####################
        
        for index in range(len(self.particles_)):
            self.particles_[index].x = self.particles_[index].x + (distance + random_normal(self.motion_distance_noise_stddev_)) * math.cos(self.particles_[index].theta)
            
            self.particles_[index].y = self.particles_[index].y + (distance + random_normal(self.motion_distance_noise_stddev_)) * math.sin(self.particles_[index].theta)
            
            self.particles_[index].theta = wrap_angle(self.particles_[index].theta + rotation + random_normal(self.motion_rotation_noise_stddev_))




        # Overwrite the previous odometry message
        self.prev_odom_msg_ = odom_msg

        # Delete any particles out of the map
        old_particles = copy.deepcopy(self.particles_)
        self.particles_ = []

        for p in old_particles:
            if not(p.x < self.map_x_min_ or p.x > self.map_x_max_ or p.y < self.map_y_min_ or p.y > self.map_y_max_):
                # Keep it
                self.particles_.append(p)

        # Normalise particle weights because particles have been deleted
        self.normalise_weights()

        # If the estimated pose is valid move it too
        if self.estimated_pose_valid_:
            estimated_pose_theta = 2. * math.acos(self.estimated_pose_.orientation.w)

            if self.estimated_pose_.orientation.z < 0.:
                estimated_pose_theta = -estimated_pose_theta

            self.estimated_pose_.position.x += math.cos(estimated_pose_theta) * distance
            self.estimated_pose_.position.y += math.sin(estimated_pose_theta) * distance

            estimated_pose_theta = wrap_angle(estimated_pose_theta + rotation)

            self.estimated_pose_.orientation.w = math.cos(estimated_pose_theta / 2.)
            self.estimated_pose_.orientation.z = math.sin(estimated_pose_theta / 2.)

        # Increment the motion update counter
        self.motion_update_count_ = self.motion_update_count_ + 1

        self.lock_.release()

    def scan_callback(self, scan_msg):
        # Receive a laser scan message

        # Only do a sensor update after num_motion_updates_
        if self.motion_update_count_ < self.num_motion_updates_:
            return

        self.lock_.acquire()

        # Determine step size (the step may not result in the correct number of rays
        step = int(math.floor(float(len(scan_msg.ranges)) / self.num_scan_rays_))

        # Setup visualisation marker
        self.marker_.points = []

        # For each particle
        first_particle = True
        for p in self.particles_:
    
            # The likelihood of the particle is the product of the likelihood of each ray
            likelihood = 1.;

            # Compare each scan ray
            for i in range(0, len(scan_msg.ranges), step): 
                # The range value from the scan message
                scan_range = scan_msg.ranges[i]

                # The angle of the ray in the frame of the robot
                local_angle = (scan_msg.angle_increment * i) + scan_msg.angle_min

                # The angle of the ray in the map frame
                global_angle = wrap_angle(p.theta + local_angle)

                # The expected range value for the particle
                particle_range = self.hit_scan(p.x, p.y, global_angle, 7.0, first_particle) #scan_msg.range_max)

                # Use "scan_range" and "particle_range" to get a likelihood
                # Multiply the ray likelihood into the "likelihood" variable
                # You will probably need "math.pi", math.sqrt()", "math.pow()", and "math.exp()"


                ####################
                ## YOUR CODE HERE ##
                ####################

                rangeDifference = (particle_range - scan_range) * (particle_range - scan_range)
                
                sensingNoiseStddev = 2*(self.sensing_noise_stddev_ * self.sensing_noise_stddev_)
                
                x = 2*math.pi*(self.sensing_noise_stddev_ * self.sensing_noise_stddev_)
                
                divisionOfRangeDiffAndSensingNoise = (rangeDifference/sensingNoiseStddev)
                
                divideOne = 1/math.sqrt(x)
                exponential = math.exp(-divisionOfRangeDiffAndSensingNoise)

                likelihood *= divideOne*exponential


            # Update the particle weight with the likelihood
            p.weight *= likelihood

            first_particle = False

        # Normalise particle weights
        self.normalise_weights()

        # Estimate the pose of the robot
        self.estimate_pose()

        self.motion_update_count_ = 0

        self.sensing_update_count_ = self.sensing_update_count_ + 1

        if self.sensing_update_count_ > self.num_sensing_updates_:
            self.resample_particles();
            self.sensing_update_count_ = 0;

        self.lock_.release()

        # Publish debug marker
        self.marker_pub_.publish(self.marker_)


def find_hit(img, p1, p2, threshold):
    # Draws a line from p1 to p2
    # Stops at the first pixel that is a "hit", i.e. above the threshold
    # Returns the pixel coordinates for the first hit

    # Extract the vector
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    step = 1.0

    dx = x2 - x1
    dy = y2 - y1
    l = math.sqrt(dx**2. + dy**2.)
    dx = dx / l
    dy = dy / l

    max_steps = int(l / step)

    for i in range(max_steps):

        # Get the next pixel
        x = int(round(x1 + dx*i))
        y = int(round(y1 + dy*i))

        # Check if it's outside the image
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return p2

        # Check for "hit"
        if img[y, x] >= threshold:
            return [x, y]

    # No hits found
    return p2


if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('particle_filter_localisation_python')

    # Create the particle filter
    ParticleFilter()

    # Loop forever while processing callbacks
    rospy.spin()
