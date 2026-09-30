TARGET      := output
EXPERIMENTS := experiments
LIB_DIR     := lib/retargeting
BUILD_DIR   := build
LIBRARY     := $(BUILD_DIR)/libretargeting.a

OPENCV_DIR := $(CURDIR)/opencv_local
CPLEX_DIR  := $(CURDIR)/cplex_local
PKG_CONFIG := PKG_CONFIG_PATH="$(OPENCV_DIR)/lib/pkgconfig:$$PKG_CONFIG_PATH" pkg-config

LIB_SRCS := $(wildcard $(LIB_DIR)/*.cpp)
LIB_OBJS := $(LIB_SRCS:$(LIB_DIR)/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ := $(BUILD_DIR)/main.o
EXP_OBJ  := $(BUILD_DIR)/experiments.o
DEPS     := $(LIB_OBJS:.o=.d) $(MAIN_OBJ:.o=.d) $(EXP_OBJ:.o=.d)

CXXFLAGS := -std=c++17 -O2 -Wall -MMD -MP -Ilib -DIL_STD \
            $(shell $(PKG_CONFIG) --cflags opencv) \
            -isystem $(CPLEX_DIR)/cplex/include \
            -isystem $(CPLEX_DIR)/concert/include
LDFLAGS  := -L$(CPLEX_DIR)/cplex/lib/x86-64_linux/static_pic \
            -L$(CPLEX_DIR)/concert/lib/x86-64_linux/static_pic
LDLIBS   := $(shell $(PKG_CONFIG) --libs opencv) \
            -lilocplex -lconcert -lcplex -lm -pthread -ldl

.PHONY: all run run-experiments clean

all: $(TARGET) $(EXPERIMENTS)

# Entry point (main.cpp) linked against the retargeting library.
$(TARGET): $(MAIN_OBJ) $(LIBRARY)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Paper experiments (ablations, timing, aspect ratios).
$(EXPERIMENTS): $(EXP_OBJ) $(LIBRARY)
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# Rebuilt from scratch: `ar rcs` alone would keep objects of deleted or renamed sources.
$(LIBRARY): $(LIB_OBJS)
	@rm -f $@
	$(AR) rcs $@ $^

$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(LIB_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	@mkdir -p $@

run: $(TARGET)
	@mkdir -p result
	@LD_LIBRARY_PATH="$(OPENCV_DIR)/lib:$$LD_LIBRARY_PATH" ./$(TARGET) $(ARGS)

run-experiments: $(EXPERIMENTS)
	@LD_LIBRARY_PATH="$(OPENCV_DIR)/lib:$$LD_LIBRARY_PATH" ./$(EXPERIMENTS) $(ARGS)

clean:
	@rm -rf $(BUILD_DIR) $(TARGET) $(EXPERIMENTS)

-include $(DEPS)
