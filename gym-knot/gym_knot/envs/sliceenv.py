import gym
from gym import spaces
import numpy as np
from gym.utils import seeding
# from ray.rllib.env.env_context import EnvContext
# from randomKnotGenerator import random_braid




class SliceEnv(gym.Env):
    """An OpenAI Gym environment for building minimal genus slice surfaces."""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, knot, lower, upper):  #, config: EnvContext):
        
        self.MP_knots={
        "K1":  [12, [1, 2, -3, 4, 5, 6, -7, -8, -7, -6, -5, -4, 3, -2, -1, -4, -3, -2, -7, 6, -9, -8, 7, 6, -5, -4, -3, -10, 11, -10, -9, -8, 7, 6, -5, -4, 6, -5, -5, 6, -7, 6, 5, -6, 8, -7, 9, 8, 7, 6, -7, 10, -9, 8, -7, 6, 5, 4, 3, 2, -6, 5, 4, 3, 7, 6, 5, 4, 5, -7, -7, 6, 7, -8, 7, 9, -11, 10, 9, 8, 7, 7, -6, -5, -7]],
        "K2":  [14, [1, 2, 3, 4, 5, -6, 7, 7, -8, 7, 6, -5, -4, -3, -2, -1, -7, -9, -8, 7, -6, 10, -9, -8, -7, 6, 5, -4, 3, 2, 4, 3, 6, 7, -6, 5, 8, 7, 6, -11, 12, -11, -10, -9, -8, -7, -6, -13, -12, -11, -10, -9, -8, 7, 6, -5, 4, 6, -5, 6, 6, -7, -6, 8, 7, 9, 8, -7, -7, 10, 9, 11, 10, 9, 12, 11, 10, 9, -8, 7, -6, 5, -4, -3, -2, -4, -3, -6, 5, 6, 7, 8, -7, -9, 8, 7, -6, -5, 4, -5, -10, 13, -12, 11, -10, 9, 8, 7, 6, -7, -7]],
        "K3":  [13, [1, 2, 3, 4, 5, 6, 7, -6, 8, 7, -6, 7, -9, 10, 9, -8, -7, 6, -5, -4, -3, -2, -1, -4, 3, -2, -5, 4, -5, -5, 6, -7, 8, 7, -6, 5, -9, -11, 10, -9, -8, -7, 6, 5, -4, 3, 5, -4, 5, 5, -6, 7, 6, -5, 8, 9, -8, -7, -6, 5, -10, 11, -10, -12, 11, -10, -9, -8, -7, -6, -5, 4, -3, 2, -3, -5, 6, 5, 4, 5, 7, 6, -5, 8, 7, 6, 9, 8, 10, 9, -8, -7, -6, 5, 6, -5, -11, 12]],
        "K4":  [10, [1, 2, 3, 4, 5, -6, 7, -6, -5, -4, -3, -2, -1, 4, 6, -8, 7, 6, -5, -4, 3, 5, -6, -5, 4, 6, 5, -6, -7, -6, -5, -4, -3, -2, -5, 6, 8, -7, 6, -5, -4, 3, 5, -4, 5, -6, 9, 8, 7, -6, 5, -6, -5, -5, 4, -3, 2, -5, 6, -7, 6, -8, -9]],
        "K5":  [13, [1, 2, 3, 4, 5, 6, 7, -6, 8, 7, 9, -8, -7, 6, 7, -6, -5, -4, -3, -2, -1, -6, -5, -4, 3, 2, -4, -3, -6, 10, -9, -8, -7, 6, -5, -4, 6, 7, -6, -5, 8, 7, 9, 8, -7, 6, -5, 6, 7, -6, 5, 5, 4, 3, -2, 5, 4, -3, -6, 5, -6, -10, -9, -8, -7, 6, -11, -10, -9, -8, 7, -6, -12, -11, -10, -9, -8, 7, 6, -5, 4, 6, 5, 6, -7, -6, 8, -7, 6, 6, 9, 8, 10, 9, 11, 10, 12, 11]],
        "H1":  [7,[-1, 2, 3, -2, 4, -5, 4, 2, -3, 1, 2, -1, -1, 4, 5, -1, -3, -4, 5, 6, -5, 4, 2, 3, -4, 5]],
        "878": [5, [-1,-2,-3,-3,-4,-2,3,1,1,-2,3,4,4,3,2,2]],
        "7_1": [2, [1,1,1,1,1,1,1]],
        "4_1": [3, [1, -2, 1, -2]],
        "7_2": [4, [1, 1, 1, 2, -1, 2, 3, -2, 3]],
        "8_1": [5, [1, 1, 2, -1, 2, 3, -2, -4, 3, -4]],
        "8_4": [5, [-1, -1, -1, 2, -1, 2, 3, -2, 3]],
        "9_1": [2, [1, 1, 1, 1, 1, 1, 1, 1, 1]],
        "9_9": [3, [1, 1, 1, 1, 1, 2, -1, 2, 2, 2]],
        "10_1": [6, [1, 1, 2, -1, 2, 3, -2, 3, 4, -3, -5, 4, -5]],
        "10_2": [3, [1, 1, 1, 1, 1, 1, 1, -2, 1, -2]],
        "11a_1": [5, [1, 1, 1, 2, -1, 2, 2, -3, 2, -3, -2, -4, 3, -4]],
        "11a_14": [4, [1, 1, -2, -2, 1, -2, 3, -2, -2, 3, 3]],
        "11a_34": [4, [1, 1, 1, -2, 1, -2, 3, -2, -2, 3, 3]],
        "11n_5": [5, [1, 1, 1, 2, -1, -3, 2, 2]],
        "11n_17": [5, [-1, -1, -2, 1 ,-2, -3, 2, -3, -2, -4, 3, -4]],
        "11n_23": [4, [1, 1, 1, 2, 1, 1, -3, 2, 2, -3, -3]],
        "11n_185": [4, [1, -2, 1, 3, -2, 1, 3, 2, 2, 2, 3]],
        "12a_4": [5, [-1, 2, -1, 2, -3, -3, 2, 2, -3, 4, -3, 4]],
        "12a_13": [5, [-1, 2, -1, 2, 3, -4, 3, 3, -4, 3, -4, -2, -2, -3]] ,
        "12n_14": [5, [-1, 2, -1, 2, -3, -3, 2, 2, 3, -4, 3, -4]],
        "12n_22": [5, [-1, 2, 3, -1, 3, -4, 2, -4, 3, -4, 3, -4]]
        }
        
        ####if "max_action_count" in config:
        ####	self.max_actions=config["max_action_count"]
        ####else: 
        ####	self.max_actions=100
        self.max_actions=200

        ''' Trying to create visualizations for the environment. '''
        self.viewer = None
        self.min_height = 30

        self.euler_lower = lower
        self.euler_upper = upper
        	
       
        ####if "starting_word" in config:
        ####	self.starting_braid=config["starting_word"]
        #### #else
        ####	self.starting_braid="random"
        self.starting_braid = knot

        
        # The maximum length a braid word can be, which is fixed once the environment is instantiated. 
        ####if "max_braid_length" in config:
        ####	self.max_braid_length=config["max_braid_length"]
        ####else: 
        ####	self.max_braid_length=175
        self.max_braid_length=175
        
        # This is the bonus that is given to the score whenever an unlinked component is created.
        ####if "bonus" in config:
        ####	self.bonus=config["bonus"]
        ####else: 
        ####	self.bonus=0
        self.bonus=0	
        	
        # The penalty given for any action which results in a reward of 0 (should be a positive value).
        ####if "inaction_penalty" in config:
        ####	self.inaction_penalty=config["inaction_penalty"]
        ####else: 
        ####	self.inaction_penalty=0.01
        self.inaction_penalty=0.01
        	
        	
        # The maximum number of strands that can be used in the braid at any given time (extra strands will be added as
        # unlinked strands on the knot component are removed).
        ####if "max_braid_index" in config:
        ####	self.index=config["max_braid_index"]
        ####else: 
        ####	self.index=25
        self.index=25
        	
        	
        	
        # The maximum number of initial strands that will be used when generating a random braid.
        ####if "max_initial_index" in config:
        ####	self.initial_index=config["max_initial_index"]
        ####else: 
        ####	self.initial_index=10
        self.initial_index=10
        	
        # A counter that is used to create new labels for components that are introduced when new strands are added 
        # (following the removal of unlinked strands on the knot component).
        self.extra_strands=0    
        
        
        ####if "final_penalty" in config:
        ####	self.final_penalty=config["final_penalty"]
        ####else: 
        ####	self.final_penalty=2*self.max_braid_length
        self.final_penalty=2*self.max_braid_length
        
        
        ####if "starting_knot_strand" in config:
        ####	self.starting_knot_strand=config["starting_knot_strand"]
        ####else: 
        ####	self.starting_knot_strand=1
        self.starting_knot_strand=1

        #Add back in when allowing MP_knots. (Not the four # though.)
        #if self.starting_braid in self.MP_knots:
        #    self.starting_word=self.MP_knots[self.starting_braid][1]
            ####elif self.starting_braid=="random":
            ####	self.starting_word=np.zeros(self.max_braid_length+1)
            ####	while len(self.starting_word)>self.max_braid_length or len(self.starting_word)==0:
            ####		self.starting_word=random_braid(max_initial_index=self.initial_index,slice_knot=True)[0]
        #else:
        # 	self.starting_word=self.starting_braid
        self.starting_word = self.starting_braid
        	
		
		# The numpy array that tracks the braid word representing the knot.
        self.word=np.array(self.starting_word) 
        assert len(self.word) <= self.max_braid_length, "Cannot initialize with braid with length longer than max_braid_length"
        

        # This list should have one entry for each strand in the braid word (self.index number of them), and tracks 
        # which strands are on the same component.  
        # For example, if the list is [1,1,2,3,3] it means the first two strands belong to component number 1 of the 
        # surface created, the third strand belongs to component number 2, while the last two strands again belong to 
        # the same component, component number 3.
        self.components=np.zeros(self.index,int)  
        self.component_count=1
        # This assigns the component of starting_knot_strand the number 1.
        self.components[self.starting_knot_strand-1]=self.component_count
        self.temp_position=len(self.word)
        # Starting with the starting_knot_strand, we trace it back through the braid word to see what other strands it 
        # connects to.
        self.next_strand=self.traceback(self.temp_position,self.starting_knot_strand)[0]
        # Label the next strand on the same component as component number 1, and assign it an Euler characteristic of 0.
        self.components[self.next_strand-1]=self.component_count
        self.eulerchar={1:0}
        # Iterate through the rest of the strands of the knot component, assigning them component number 1.
        while self.next_strand!=self.starting_knot_strand:
            self.next_strand=self.traceback(self.temp_position,self.next_strand)[0]
            self.components[self.next_strand-1]=self.component_count
        # Iterate now through the remaining strands, assigning increasing values for each subsequent component.
        for jjj in range(self.index):
            if self.components[jjj]==0:
                self.component_count+=1
                self.new_starting_knot_strand=jjj+1
                self.components[self.new_starting_knot_strand-1]=self.component_count
                self.next_strand=self.traceback(self.temp_position,self.new_starting_knot_strand)[0]
                self.components[self.next_strand-1]=self.component_count
                while self.next_strand!=self.new_starting_knot_strand:
                    self.next_strand=self.traceback(self.temp_position,self.next_strand)[0]
                    self.components[self.next_strand-1]=self.component_count   
        # Assign an euler characteristic of 1 to any components other than the knot component.
        for key in self.components:
            if key!=1:
                self.eulerchar[key]=1
        # Initiate the score to 0.
        self.score=0
        # Initiate the cursor position.
        self.cursor=np.array([0,1])
        # Run through the different strands.  If any strands corresponding to component number 1 are unlinked, delete them.
        for jjj in range(self.index):
            self.unlinked_strand_check(jjj+1)
        #self.state_tuple = self.get_state_tuple()
        self.encoded_state_length=len(self.encode_state())
        self.action_map={0: "Remove Crosing",
                         1: "Move Down",
                         2: "Move Up",
                         3: "Move Left",
                         4: "Move Right",
                         5: "Cut",
                         6: "Add Positive r2",
                         7: "Add Negative r2",
                         8: "Remove r2",
                         9: "r3",
                         10: "Far comm",
                         11: "Add Positive crossing",
                         12: "Add Negative crossing"}
        self.inverse_action_map={"Remove Crossing": 0,
                                 "Move Down": 1,
                                 "Move Up": 2,
                                 "Move Left": 3,
                                 "Move Right": 4,
                                 "Cut": 5,
                                 "Add Positive r2": 6,
                                 "Add Negative r2": 7,
                                 "Remove r2": 8,
                                 "r3": 9,
                                 "Far comm": 10,
                                 "Add Positive crossing": 11,
                                 "Add Negative crossing": 12}
        # Define the action space for Gym, the list of integers from 0 to 12 inclusive.
        self.action_space = spaces.Discrete(13)
        # Define the lower and upper bounds for the observations space.
        self.low_bound=np.array([-self.index+1 for jjj in range(self.max_braid_length)]+[0,1]+[1 for jjj in range(self.index)]+[-np.inf for jjj in range(self.index)])
        self.high_bound=np.array([self.index-1 for jjj in range(self.max_braid_length)]+[self.max_braid_length,self.index]+[self.index for jjj in range(self.index)]+[1 for jjj in range(self.index)])
        self.observation_space=spaces.Box(low=self.low_bound,high=self.high_bound,dtype=np.int64)
        self.action_count=0
        self.seed()
        self.done=False
        self.action_list={}
        metadata = {"render.modes": ["human"]}
        

    # Takes as input a required position (corresponding to a letter in the braid word, indexed starting at 0), and an 
    # optional pair of strands (if the strands are ommited, the two strands involved in the crossing at the given 
    # position are chosen).  The strands are numbered from 1 to self.index.  Returns the position of the strands once 
    # they have been traced back up to the top of the braid word.
    def traceback(self,position,strand1=-1,strand2=-1):
        if len(self.word)==0:
            return strand1,strand2
        if strand1+strand2==-2:
            strand1=np.abs(self.word[position])
            strand2=np.abs(self.word[position])+1
        for jjj in range(position):
            if np.abs(self.word[position-1-jjj])==strand1:
                strand1+=1
            elif np.abs(self.word[position-1-jjj])==strand1-1:
                strand1-=1
            if np.abs(self.word[position-1-jjj])==strand2:
                strand2+=1
            elif np.abs(self.word[position-1-jjj])==strand2-1:
                strand2-=1
        return strand1,strand2 
    
    # Takes as input a strand number.  If it corresponds to a component other than component 1 it returns immediately.  
    # Otherwise it runs through the length of the braid word, checking to see if that component is involved in any of 
    # the crossings in the braid word.  If it is, it returns having done nothing.  If the given strand is not involved
    # in any of the crossings, then it increments the euler characteristic of component 1, adds 1+self.bonus to the 
    # value of self.score.  It then deletes that strand from self.components, adds an extra strand with Euler 
    # characteristic 1, and changes all of the crossings in the braid words to account for the removal of the unlinked
    # strand.  STRANDS ARE NUMBERED AS USUAL IN BRAID NOTATION STARTING AT 1, NOT USING PYTHON ARRAY NOTATION 
    # STARTING AT 0.
    def unlinked_strand_check(self,strand):
        # If the strand corresponds to a component other than component 1 it returns immediately.
        if self.components[strand-1]!=1:
            return
        # If the strand corresponds to component number 1, then run through the length of the braid word, checking to 
        # see if that strand is involved in any of the crossings in the braid word.  If it is, it returns having done 
        # nothing.
        for jjj in range(len(self.word)):
            if np.abs(self.word[jjj])==strand-1 or np.abs(self.word[jjj])==strand:
                return
        # If the given strand is not involved in any of the crossings, increment the Euler characteristic by one, and 
        # change the score by one plus the value of self.bonus.
        self.eulerchar[1]+=1
        self.score+=self.bonus+1
        # Delete the component corresponding to the unlinked strand that is being deleted.
        self.components=np.delete(self.components,strand-1)
        # Add a new strand to the component list with new component number, and set its Euler characteristic to one.
        self.components=np.append(self.components,self.index+1+self.extra_strands)
        self.extra_strands+=1
        self.eulerchar[self.components[-1]]=1
        # Change the crossing number of all letters with crossing number greater than or equal to the deleted strand
        # number.  
        for jjj in range(len(self.word)):
            if self.word[jjj]>=strand+1:
                self.word[jjj]-=1
            if -self.word[jjj]>=strand+1:
                self.word[jjj]+=1
#####        # Shift the cursor position to the left one space if it sits to the right of the strand being deleted.
#####        #if self.cursor[1]>strand:
#####            #self.cursor[1]-=1

    def arrange_components(self):
        current_comp=-2
        temp_eulerlist={1:self.eulerchar[1]}
        complist=self.components
        for lll in complist:
            if lll>1:
                for www in np.where(complist==lll)[0]:
                    complist[www]=current_comp
                temp_eulerlist[np.abs(current_comp)]=self.eulerchar[lll]
                current_comp-=1
        self.eulerchar=temp_eulerlist
        self.components=np.abs(complist) 
                
    # Moves the cursor position up (thinking of the braid as written from top to bottom).
    def move_up(self):
        if self.cursor[0]>=1:
            self.cursor[0]-=1
        else:
            self.cursor[0]=len(self.word)
            
    # Moves the cursor position down (thinking of the braid as written from top to bottom).        
    def move_down(self):
        if self.cursor[0]<=len(self.word)-1:
            self.cursor[0]+=1
        else:
            self.cursor[0]=0
            
    # Moves the cursor position right (thinking of the strands as being written left to right starting at 1).
    def move_right(self):
        if self.cursor[1]<=self.index-2:
            self.cursor[1]+=1
        else:
            self.cursor[1]=1
            
    # Moves the cursor position left (thinking of the strands as being written left to right starting at 1).        
    def move_left(self):
        if self.cursor[1]>=2:
            self.cursor[1]-=1
        else:
            self.cursor[1]=self.index-1
    
    # Cuts the braid at the position in the cursor, and moves the subword before the cursor to the end of the braid.
    # It also modifies the component list appropriately, before setting the position of the cursor to the beginning of
    # the new braid word.
    def cut(self):
        for iii in range(self.cursor[0]):
            a=(self.components[np.abs(self.word[iii])-1])
            b=(self.components[np.abs(self.word[iii])])
            self.components[np.abs(self.word[iii])-1]=b
            self.components[np.abs(self.word[iii])]=a
        self.word=np.concatenate((self.word[self.cursor[0]:],self.word[:self.cursor[0]]))
        self.cursor[0]=0
        self.arrange_components()
     
    # Inserts a pair of cancelling crossings (positive, then negative) to the braid word at the location specified by
    # the cursor.  If the resulting braid word would have length larger than self.max_braid_length, then it does
    # nothing.
    def r2_add_pos(self):
        if len(self.word)>=self.max_braid_length-1:
            return
        self.word=np.concatenate((self.word[:self.cursor[0]],np.array([self.cursor[1],-self.cursor[1]]),self.word[self.cursor[0]:]))
        self.arrange_components()
    
    # Inserts a pair of cancelling crossings (negative, then positive) to the braid word at the location specified by
    # the cursor.  If the resulting braid word would have length larger than self.max_braid_length, then it does
    # nothing.
    def r2_add_neg(self):
        if len(self.word)>=self.max_braid_length-1:
            return
        self.word=np.concatenate((self.word[:self.cursor[0]],np.array([-self.cursor[1],self.cursor[1]]),self.word[self.cursor[0]:]))
        self.arrange_components()
    
    
    # Starts scanning the braid word at the position specified by the cursor, then removes the first instance of a pair
    # of letters of the form (a,-a).  If none exist, no action is taken.
    def r2_rm(self):
        for iii in range(len(self.word)):
            if self.word[(iii+self.cursor[0])%len(self.word)]==-self.word[(iii+self.cursor[0]+1)%len(self.word)]:
                left_strand=np.abs(self.word[(iii+self.cursor[0])%len(self.word)]) 
                right_strand=left_strand+1
                strand1,strand2=self.traceback((self.cursor[0]+iii)%len(self.word),left_strand,right_strand)
                # If the two crossings to be removed are at the first and last position of the braid, the components of
                # the corresponding strands must be swapped.  
                if (iii+self.cursor[0]+1)%len(self.word)==0:
                    a=self.components[left_strand-1]
                    b=self.components[right_strand-1]
                    self.components[left_strand-1]=b
                    self.components[right_strand-1]=a
                # Deletes the corresponding crossings from the braid words.
                loc1=max((iii+self.cursor[0])%len(self.word),(iii+self.cursor[0]+1)%len(self.word))
                loc2=min((iii+self.cursor[0])%len(self.word),(iii+self.cursor[0]+1)%len(self.word))
                self.word=np.delete(self.word,loc1)
                self.word=np.delete(self.word,loc2)
                ##########
                ###### Check the strands involved in the cancelling pair to see if they are now unlinked (checking the
                ###### larger of the two first, so that if it is deleted it doesn't mess up the number of the other one).
                #####for jjj in np.where(self.components==1)[0]:
                    #####self.unlinked_strand_check(jjj+1)
                ##########
                # Check for unlinked strands of component number 1, starting at the right and moving left so as to not miss 
                # any strands from component number 1.
                comp_1_list=np.where(self.components==1)[0]
                for jjj in range(len(comp_1_list)):
                    self.unlinked_strand_check(comp_1_list[-jjj-1]+1)
                #####self.unlinked_strand_check(max(strand1,strand2))
                #####self.unlinked_strand_check(min(strand1,strand2))
                # If the cursor position is now outside the length of the word, set it to the beginning of the word.
                if self.cursor[0]>len(self.word):
                    self.cursor[0]=len(self.word)
                break
        self.arrange_components()
    ######## FIGURE OUT WHERE TO CHANGE CURSOR POSITION   
    
    
    
    # Start scanning the braid word at the position specified by the cursor, then apply a Reidemeister 3 move at the 
    # position it's possible to.
    def r3(self):
        for iii in range(len(self.word)):
            loc1=(iii+self.cursor[0])%len(self.word)
            loc2=(iii+self.cursor[0]+1)%len(self.word)
            loc3=(iii+self.cursor[0]+2)%len(self.word)
            if self.word[loc1]==self.word[loc3] and np.abs(self.word[loc1]-self.word[loc2])==1:
                if (iii+self.cursor[0]+1)%len(self.word)==0 or (iii+self.cursor[0]+2)%len(self.word)==0:
                    eps=np.abs(self.word[loc2])-np.abs(self.word[loc1])
                    if eps==1:
                        left_strand=np.abs(self.word[loc1])
                        middle_strand=left_strand+1
                        right_strand=left_strand+2
                        a=self.components[left_strand-1]
                        b=self.components[middle_strand-1]
                        c=self.components[right_strand-1]
                        self.components[left_strand-1]=b
                        self.components[middle_strand-1]=c
                        self.components[right_strand-1]=a
                    elif eps==-1:
                        middle_strand=np.abs(self.word[loc1])
                        left_strand=middle_strand-1
                        right_strand=middle_strand+1
                        a=self.components[left_strand-1]
                        b=self.components[middle_strand-1]
                        c=self.components[right_strand-1]
                        self.components[left_strand-1]=c
                        self.components[middle_strand-1]=a
                        self.components[right_strand-1]=b
                letter1=(self.word[loc1])
                letter2=(self.word[loc2])
                self.word[loc1]=letter2
                self.word[loc3]=letter2
                self.word[loc2]=letter1
                break
            elif self.word[loc1]==-self.word[loc3] and np.abs(np.abs(self.word[loc1])-np.abs(self.word[loc2]))==1:
                if (iii+self.cursor[0]+1)%len(self.word)==0 or (iii+self.cursor[0]+2)%len(self.word)==0:
                    eps=np.abs(self.word[loc2])-np.abs(self.word[loc1])
                    if eps==1:
                        left_strand=np.abs(self.word[loc1])
                        middle_strand=left_strand+1
                        right_strand=left_strand+2
                        a=self.components[left_strand-1]
                        b=self.components[middle_strand-1]
                        c=self.components[right_strand-1]
                        self.components[left_strand-1]=b
                        self.components[middle_strand-1]=c
                        self.components[right_strand-1]=a
                    elif eps==-1:
                        middle_strand=np.abs(self.word[loc1])
                        left_strand=middle_strand-1
                        right_strand=middle_strand+1
                        a=self.components[left_strand-1]
                        b=self.components[middle_strand-1]
                        c=self.components[right_strand-1]
                        self.components[left_strand-1]=c
                        self.components[middle_strand-1]=a
                        self.components[right_strand-1]=b
                letter1=(self.word[loc1])
                letter2=(self.word[loc2])
                letter3=(self.word[loc3])
                sign=np.sign(letter1)*np.sign(letter2)
                self.word[loc1]=np.abs(letter2)*np.sign(letter3)
                self.word[loc2]=sign*letter1
                self.word[loc3]=sign*letter2
                break
        self.arrange_components()
    ######## FIGURE OUT WHERE TO CHANGE CURSOR POSITION           
    
     
    # Start scanning the braid word at the position specified by the cursor, then swap the first two letters which 
    # in absolute value by two or greater.
    def far_comm(self):
        for iii in range(len(self.word)):
            loc1=(iii+self.cursor[0])%len(self.word)
            loc2=(iii+self.cursor[0]+1)%len(self.word)
            # If the two crossings to be swapped are at the beggining and end of the word respectively, then we alter
            # the component list accordingly.  
            if np.abs(np.abs(self.word[loc1])-np.abs(self.word[loc2]))>=2:
                if (iii+self.cursor[0]+1)%len(self.word)==0:
                    left_strand1=np.abs(self.word[loc1])
                    right_strand1=left_strand1+1
                    left_strand2=np.abs(self.word[loc2])
                    right_strand2=left_strand2+1
                    a1=self.components[left_strand1-1]
                    b1=self.components[right_strand1-1]
                    a2=self.components[left_strand2-1]
                    b2=self.components[right_strand2-1]
                    self.components[left_strand1-1]=b1
                    self.components[right_strand1-1]=a1
                    self.components[left_strand2-1]=b2
                    self.components[right_strand2-1]=a2
                letter1=self.word[loc1]
                letter2=self.word[loc2]
                self.word[loc1]=letter2
                self.word[loc2]=letter1
                break
        self.arrange_components()
                
    # Adds a positive crossing at the position specified by the cursor.  If the length of the resulting word would
    # exceed self.max_braid_length then nothing is done.
    def add_crossing_pos(self):
        if len(self.word)>=self.max_braid_length:
            return
        self.word=np.concatenate((self.word[:self.cursor[0]],[self.cursor[1]],self.word[self.cursor[0]:]))
        # Change the components of the corresponding strands to make them both the value of the smaller of the two 
        # components.
        strand1,strand2=self.traceback(self.cursor[0])
        comp1=self.components[strand1-1]
        comp2=self.components[strand2-1]
        mincomp=min(comp1,comp2)
        maxcomp=max(comp1,comp2)
        for jjj in range(len(self.components)):
            if self.components[jjj]==comp1 or self.components[jjj]==comp2:
                self.components[jjj]=mincomp
        # If the components of the corresponding strand were already the same, then the Euler characteristic decreases
        # by one.
        if comp1==comp2:
            self.eulerchar[mincomp]-=1
            if mincomp==1:
                self.score-=1
        # If the components of the corresponding strands were not equal, then the Euler characteristic of the resulting
        # component is the sum of the Euler characteristics of the two components minus one.
        else:
            self.eulerchar[mincomp]=self.eulerchar[comp1]+self.eulerchar[comp2]-1
            # If the smaller of the two components was component number 1, we update the score along with the Euler
            # characteristic of the strand.
            if mincomp==1:
                self.score=self.score+self.eulerchar[maxcomp]-1
            # Delete the record of the Euler characteristic of the larger of the two components.
            del self.eulerchar[maxcomp]
        # Check for unlinked strands of component number 1, starting at the right and moving left so as to not miss 
        # any strands from component number 1.
        comp_1_list=np.where(self.components==1)[0]
        for jjj in range(len(comp_1_list)):
            self.unlinked_strand_check(comp_1_list[-jjj-1]+1)
        self.arrange_components()
            
    # Adds a negative crossing at the position specified by the cursor. If the length of the resulting word would
    # exceed self.max_braid_length then nothing is done.           
    def add_crossing_neg(self):
        if len(self.word)>=self.max_braid_length:
            return
        self.word=np.concatenate((self.word[:self.cursor[0]],[-self.cursor[1]],self.word[self.cursor[0]:]))
        # Change the components of the corresponding strands to make them both the value of the smaller of the two 
        # components.
        strand1,strand2=self.traceback(self.cursor[0])
        comp1=self.components[strand1-1]
        comp2=self.components[strand2-1]
        mincomp=min(comp1,comp2)
        maxcomp=max(comp1,comp2)
        for jjj in range(len(self.components)):
            if self.components[jjj]==comp1 or self.components[jjj]==comp2:
                self.components[jjj]=mincomp
        # If the components of the corresponding strand were already the same, then the Euler characteristic decreases
        # by one.
        if comp1==comp2:
            self.eulerchar[mincomp]-=1
            if mincomp==1:
                self.score-=1
        # If the components of the corresponding strands were not equal, then the Euler characteristic of the resulting
        # component is the sum of the Euler characteristics of the two components minus one.
        else:
            self.eulerchar[mincomp]=self.eulerchar[comp1]+self.eulerchar[comp2]-1
            # If the smaller of the two components was component number 1, we update the score along with the Euler
            # characteristic of the strand.
            if mincomp==1:
                self.score=self.score+self.eulerchar[maxcomp]-1
            # Delete the record of the Euler characteristic of the larger of the two components.
            del self.eulerchar[maxcomp]
        # Check for unlinked strands of component number 1, starting at the right and moving left so as to not miss 
        # any strands from component number 1.
        comp_1_list=np.where(self.components==1)[0]
        for jjj in range(len(comp_1_list)):
            self.unlinked_strand_check(comp_1_list[-jjj-1]+1)
        self.arrange_components()
            
            
    # Remove the crossing at the position specified by the cursor.  
    def rm_crossing(self):
        if len(self.word)==0:
            return
        if self.cursor[0]==len(self.word):
            self.cursor[0]=0
        # Change the components of the corresponding strands to make them both the value of the smaller of the two 
        # components.
        strand1,strand2=self.traceback(self.cursor[0])
        self.word=np.delete(self.word,self.cursor[0])
        comp1=self.components[strand1-1]
        comp2=self.components[strand2-1]
        mincomp=min(comp1,comp2)
        maxcomp=max(comp1,comp2)
        for jjj in range(len(self.components)):
            if self.components[jjj]==comp1 or self.components[jjj]==comp2:
                self.components[jjj]=mincomp
        # If the components of the corresponding strand were already the same, then the Euler characteristic decreases
        # by one.
        if comp1==comp2:
            self.eulerchar[mincomp]-=1
            if mincomp==1:
                self.score-=1
        # If the components of the corresponding strands were not equal, then the Euler characteristic of the resulting
        # component is the sum of the Euler characteristics of the two components minus one.
        else:
            self.eulerchar[mincomp]=self.eulerchar[comp1]+self.eulerchar[comp2]-1
            # If the smaller of the two components was component number 1, we update the score along with the Euler
            # characteristic of the strand.
            if mincomp==1:
                self.score=self.score+self.eulerchar[maxcomp]-1
            # Delete the record of the Euler characteristic of the larger of the two components.
            del self.eulerchar[maxcomp]
        # Check for unlinked strands of component number 1, starting at the right and moving left so as to not miss 
        # any strands from component number 1.
        comp_1_list=np.where(self.components==1)[0]
        for jjj in range(len(comp_1_list)):
            self.unlinked_strand_check(comp_1_list[-jjj-1]+1)
        #####self.unlinked_strand_check(max(strand1,strand2))
        #####self.unlinked_strand_check(min(strand1,strand2))
        self.arrange_components()
######## FIGURE OUT WHERE TO CHANGE CURSOR POSITION        
        
    
    # Check if the braid is in a terminal state.  It is a terminal state if it has length zero or if none of the 
    # strand correspond to component number one.  
    def is_Terminal(self):
        # Check if the braid word has length 0, if so, return True.
        if len(self.word)==0:
            #print('in is_Terminal if len(self.word) == 0')
            #print('checking is_Euler')
            #print(f'see if {self.euler_lower} <= {self.eulerchar[1]} <= {self.euler_upper}') 
            '''Add is_Euler function call here.'''
            return self.is_Euler()
        # Check if there are any strands with component number one, if so, return False.  
        for jjj in range(len(self.components)):
            if self.components[jjj]==1:
                return False
        '''Add is_Euler function call here.'''
        return self.is_Euler()

    '''Add is_Euler function here. Make sure the self.eulchar[1] is between 1-2g for the minimal and maximal ones.'''
    def is_Euler(self):
        # Check if self.eulerchar[1] is between the lower and upper bound for the knot 
        if self.euler_lower <= self.eulerchar[1] <= self.euler_upper:
            return True
        else:
            return False
        
    
    # Function to display information of the current state of the braid word.
    def info(self):
        print("Knot:\t\t\t", self.starting_braid)
        print("Braid word:\t\t",self.word)
        print("Component list:\t\t",self.components)
        print("Euler characteristics:\t",self.eulerchar)
        print("Score:\t\t\t",self.score)
        print("Cursor:\t\t\t",self.cursor)
        print("Is Terminal:\t\t",self.is_Terminal())
    
  

    # One-hot encodes the cursor position and the braid word.  
    def one_hot(self):
        # Initializes the one-hot array.
        self.one_hot_word=0.25*np.ones(2*(self.index-1)*(self.max_braid_length)+(self.max_braid_length+1)+(self.index-1))
        # Encode the cursor position.
        self.one_hot_word[self.cursor[0]]=0.75
        self.one_hot_word[self.cursor[1]+self.max_braid_length]=0.75
        self.offset=self.max_braid_length+self.index-1
        # Encode the braid word.
        for jjj in range(len(self.word)):
            self.one_hot_word[self.offset+jjj*2*(self.index-1)+(np.sign(self.word[jjj])+1)//2*self.word[jjj]-(np.sign(self.word[jjj])-1)//2*(np.abs(self.word[jjj])+self.index-1)]=0.75
        return self.one_hot_word
    
    # Combines the above one-hot encoding with a one-hot encoding of the component list and euler characteristic 
    # (though it does not capture any component numbers larger than the index, which could have been added later in the
    # process).
    def full_one_hot(self):
        self.ohmat=0.25*np.ones((self.index+1,self.index))
        for jjj in range(self.index):
            if self.components[jjj]<=self.index:
                self.ohmat[jjj,self.components[jjj]-1]=0.75
            self.ohmat[self.index,jjj]=self.eulerchar[self.components[jjj]]
        foh=np.concatenate((self.one_hot(),np.reshape(self.ohmat,self.index*(self.index+1))))
        return foh
    
    def print_braid(self):
        down_arrow = "\u2193"
        right_arrow = "\u2192"
        row = self.cursor[0]
        column = self.cursor[1]
        if len(self.word) == 0:
            print(" ", end="")
            print(" "*(2*column-1), end='')
            print(down_arrow)
            print(" ", end="") 
            print("| "*(self.index))           
            print(right_arrow, end='')
            print("| "*(self.index))
            return
        print(" "*(2*column-1)+" ", end='')
        print(down_arrow)
        print(" ", end="")
        print("| "*self.index)
        i = 1
        for cross in self.word:
            if i == row+1:
                print(right_arrow, end="")
            else:
                print(" ", end="")
            print("| "*(abs(cross)-1), end='')
            if cross > 0:
                print(" /  ", end='')
            else:
                print(" \  ", end='')
            print("| "*(self.index - abs(cross) - 1))
            i += 1
        if i == row+1:
            print(right_arrow, end="")
        else:
            print(" ", end="")
        print("| "*self.index)
    
    def print_action_sequence(self, action_list, gamma=0.99):
        self.info()
        self.print_braid()
        reward_seq=[]
        for action in action_list:
            reward, _, _ = self.action(action)
            reward_seq.append(reward)
            print("="*60)
            print("Action {}: {}".format(action, self.action_map[action]))
            print("Reward: {}".format(reward))
            print("="*60)
            self.info()
            self.print_braid()
    
    def encode_state(self, zero=0, one=1, display=False):
        """updated encode_state() function. Uses lists instead of numpy arrays. New implementation
        is 3-4 times faster.
        By Spencer
        Encodes our state for input into a neural network
        The braid, component list, and cursor positions are one-hot-encoded while the Euler 
        components are simply put in since they are unbounded.
        braid encoding"""
        encoded=[]
        braid_encoding=[]
        #padded zeros encoding
        for i in range(self.max_braid_length-len(self.word)):
            code=[zero for i in range(2*(self.index)-1)]
            index=self.index-1
            code[index]=one
            braid_encoding+=code
        #crossings encoding
        for crossing in self.word:
            code=[zero for i in range(2*(self.index)-1)]
            index=self.index-1+crossing
            code[index]=one
            braid_encoding+=code
        
        encoded+=braid_encoding
        comp_encoding=[]
        #component list encoding
        for component in self.components:
            code=[zero for i in range(self.index+1)]
            code[component-1]=one
            comp_encoding+=code
        encoded+=comp_encoding
        euler_encoding=[]
        #Euler list encoding
        code=[zero for i in range(len(self.components))]
        try:
            for i in range(len(self.components)):
                code[i]=self.eulerchar[self.components[i]]
        except KeyError:
            print("Key Error: {}".format(self.components[i]))
            print("Components: {}".format(self.components))
            print("Eulerchar: {}".format(self.eulerchar))
        euler_encoding+=code     
        encoded+=euler_encoding
        #Cursor position encoding
        #row cursor encoding
        cursor_encoding=[]
        code=[zero for i in range(self.max_braid_length+1)]
        index=self.cursor[0]
        code[index]=one
        cursor_encoding+=code
        #column cursor encoding
        code=[zero for i in range(self.index-1)]
        index=self.cursor[1]-1
        code[index]=one
        cursor_encoding+=code
        
        encoded+=cursor_encoding
        if display:
            line_length=100
            print("="*line_length)
            print("State Encoding")
            self.info()
            print("Braid encoding length: {}".format(len(braid_encoding)))
            print("Braid encoding: {}\n".format(braid_encoding))
            print("Component encoding length: {}".format(len(comp_encoding)))
            print("Component encoding: {}\n".format(comp_encoding))
            print("Euler encoding length: {}".format(len(euler_encoding)))
            print("Euler encoding: {}\n".format(euler_encoding))
            print("Cursor encoding length: {}".format(len(cursor_encoding)))
            print("Cursor encoding: {}]\n".format(cursor_encoding))
            print("Full encoding length: {}".format(len(encoded)))
            print("Full encoding: {}\n".format(encoded)) 
            print("="*line_length)
        return np.array(encoded)
    
    def complete_state(self):
        """ Gathers all state information into a single vector, without using one hot encoding. """
        state=list(self.word)+[0 for jjj in range(self.max_braid_length-len(self.word))]+list(self.cursor)+list(self.components)+[self.eulerchar[jjj] for jjj in self.components]
        return state
    
    
    def reset(self):
        if self.starting_braid=="random":
            self.starting_word=np.zeros(self.max_braid_length+1)
            while len(self.starting_word)>self.max_braid_length or len(self.starting_word)==0:
                self.starting_word=random_braid(max_initial_index=self.initial_index,slice_knot=True)[0]
        # The numpy array that tracks the braid word representing the knot.
        self.word=np.array(self.starting_word)
        # A counter that is used to create new labels for components that are introduced when new strands are added 
        # (following the removal of unlinked strands on the knot component).
        self.extra_strands=0
        # This list should have one entry for each strand in the braid word (self.index number of them), and tracks 
        # which strands are on the same component.  
        # For example, if the list is [1,1,2,3,3] it means the first two strands belong to component number 1 of the 
        # surface created, the third strand belongs to component number 2, while the last two strands again belong to 
        # the same component, component number 3.
        self.components=np.zeros(self.index,int)  
        self.component_count=1
        # This assigns the component of starting_knot_strand the number 1.
        self.components[self.starting_knot_strand-1]=self.component_count
        self.temp_position=len(self.word)
        # Starting with the starting_knot_strand, we trace it back through the braid word to see what other strands it 
        # connects to.
        self.next_strand=self.traceback(self.temp_position,self.starting_knot_strand)[0]
        # Label the next strand on the same component as component number 1, and assign it an Euler characteristic of 0.
        self.components[self.next_strand-1]=self.component_count
        self.eulerchar={1:0}
        # Iterate through the rest of the strands of the knot component, assigning them component number 1.
        while self.next_strand!=self.starting_knot_strand:
            self.next_strand=self.traceback(self.temp_position,self.next_strand)[0]
            self.components[self.next_strand-1]=self.component_count
        # Iterate now through the remaining strands, assigning increasing values for each subsequent component.
        for jjj in range(self.index):
            if self.components[jjj]==0:
                self.component_count+=1
                self.new_starting_knot_strand=jjj+1
                self.components[self.new_starting_knot_strand-1]=self.component_count
                self.next_strand=self.traceback(self.temp_position,self.new_starting_knot_strand)[0]
                self.components[self.next_strand-1]=self.component_count
                while self.next_strand!=self.new_starting_knot_strand:
                    self.next_strand=self.traceback(self.temp_position,self.next_strand)[0]
                    self.components[self.next_strand-1]=self.component_count   
        # Assign an euler characteristic of 1 to any components other than the knot component.
        for key in self.components:
            if key!=1:
                self.eulerchar[key]=1
        # Initiate the score to 0.
        self.score=0
        # Initiate the cursor position.
        self.cursor=np.array([0,1])
        # Run through the different strands.  If any strands corresponding to component number 1 are unlinked, delete them.
        for jjj in range(self.index):
            self.unlinked_strand_check(jjj+1)
        #self.state_tuple = self.get_state_tuple()
        
        # Define the action space for Gym, the list of integers from 0 to 12 inclusive.
        self.action_space = spaces.Discrete(13)
        # Define the lower and upper bounds for the observations space.
        self.low_bound=np.array([-self.index+1 for jjj in range(self.max_braid_length)]+[0,1]+[1 for jjj in range(self.index)]+[-np.inf for jjj in range(self.index)])
        self.high_bound=np.array([self.index-1 for jjj in range(self.max_braid_length)]+[self.max_braid_length,self.index]+[self.index for jjj in range(self.index)]+[1 for jjj in range(self.index)])
        self.observation_space=spaces.Box(low=self.low_bound,high=self.high_bound,dtype=np.int64)
        self.action_count=0
        self.seed()
        self.done=False
        self.action_list={}
        #self.max_actions=config["max_action_count"]
        #self.max_actions=20
        return self.complete_state()
    
    
    # Associating numbers 0 through 13 to the braid word actions defined above.
    def step(self, action_number):
        """Associating numbers 0 through 13 to the braid word actions defined above.
        This is also where the reward function for the MDP is implemented"""
        old_score=self.eulerchar[1]
        # Use the self.info variable to keep track of actions we have performed this episode.
        self.action_list[self.action_count]=action_number
        if action_number==1:
            self.move_down()
        elif action_number==2:
            self.move_up()
        elif action_number==3:
            self.move_left()
        elif action_number==4:
            self.move_right()
        elif action_number==5:
            self.cut()
        elif action_number==6:
            self.r2_add_pos()
        elif action_number==7:
            self.r2_add_neg()
        elif action_number==8:
            self.r2_rm()
        elif action_number==9:
            self.r3()
        elif action_number==10:
            self.far_comm()
        elif action_number==11:
            self.add_crossing_pos()
        elif action_number==12:
            self.add_crossing_neg()
        elif action_number==0:
            self.rm_crossing()
        else:
            assert True==False, "Invalid action passed: {}".format(action_number)
        for component in self.components:
            assert component > 0, "Error"
        state=self.complete_state()
        reward=-self.inaction_penalty+self.eulerchar[1]-old_score
        self.done=self.is_Terminal()
        self.action_count+=1
        if self.action_count==self.max_actions:
            self.done=True
            reward-=self.final_penalty
        for component in self.components:
            assert component in self.eulerchar.keys(), "Components and Eulerchar have become misaligned. Components: {} Eulerchar: {}".format(self.components, self.eulerchar)
        return [state, reward, self.done, self.action_list]  
    
    
    def render(self,mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = 2.6 * 2
        scale = screen_width / world_width

        #if self.viewer is None:
        #    from gym.envs.classic_control import rendering
        #    self.print = self.print_braid()
        #    self.viewer = rendering.Viewer(screen_width, screen_height)
        #    self.viewer.add_geom(self.print)
        #    self.info()
        #    print("Action list = ",self.action_list)
        #    self.print_braid()
        #    self.add
        #return self.viewer.render(return_rgb_array=mode == "rgb_array")
        #self.info()
        #self.print_action_sequence(self, )
        self.info()
        self.print_braid()
        
        
    def seed(self,seed=None):
        # For instructions see https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
        

            
