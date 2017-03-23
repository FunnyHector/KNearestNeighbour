class KNearestNeighbourClassifier
  FLOAT_TOLERANCE = 0.00000001

  attr_reader :k_value, :sepal_length_range, :sepal_width_range, :petal_length_range, :petal_width_range

  def initialize(training_file, test_file, k_value, num_clusters, do_cluster)
    @training_file = training_file
    @test_file     = test_file
    @k_value       = k_value
    @num_clusters  = num_clusters
    @do_cluster    = do_cluster
  end

  def read_data
    read_training_data
    read_test_data
  end

  def estimate_ranges
    all_instances = @training_set + @test_set

    sepal_length_collection = all_instances.map(&:sepal_length)
    sepal_width_collection  = all_instances.map(&:sepal_width)
    petal_length_collection = all_instances.map(&:petal_length)
    petal_width_collection  = all_instances.map(&:petal_width)

    @sepal_length_range = sepal_length_collection.max - sepal_length_collection.min
    @sepal_width_range  = sepal_width_collection.max - sepal_width_collection.min
    @petal_length_range = petal_length_collection.max - petal_length_collection.min
    @petal_width_range  = petal_width_collection.max - petal_width_collection.min
  end

  def cluster_training_set
    initialise_clusters
    converge_clusters
    label_training_set_by_clusters
  end

  def classify_test_set
    @test_set.each { |instance| classify!(instance) }
  end

  def show_result
    result = "========== Results of classification (test set) ==========\n"
    result << ["No.", "sepal_length", "sepal_width", "petal_length", "petal_width", "label", "classified_class"].join("  ") << "\n"

    @test_set.each_with_index do |instance, index|
      result << ["%02d" % (index + 1), "#{instance.sepal_length}", "#{instance.sepal_width}", "#{instance.petal_length}", "#{instance.petal_width}", "#{instance.label}", "#{instance.classified_class}"].join("  ")
      result << "    # classification mismatched" if instance.classified_class_mismatched?
      result << "\n"
    end

    if @do_cluster
      result << "\n========== Results of clustering (training set) ==========\n"
      result << ["No.", "sepal_length", "sepal_width", "petal_length", "petal_width", "label", "clustered_class"].join("  ") << "\n"

      @training_set.each_with_index do |instance, index|
        result << ["%02d" % (index + 1), "#{instance.sepal_length}", "#{instance.sepal_width}", "#{instance.petal_length}", "#{instance.petal_width}", "#{instance.label}", "#{instance.clustered_class}"].join("  ")
        result << "\n"
      end
    end

    # some key figures for summary
    num_of_mismatch = @test_set.count(&:classified_class_mismatched?)
    test_set_size   = @test_set.size
    accuracy        = ((test_set_size - num_of_mismatch).to_f / test_set_size * 100).round(2)

    summary = "".tap do |str|
      str << "\n======================= Summary =======================\n"
      str << "K Value: #{k_value}\n"
      str << "Training data size: #{@training_set.size}\n"
      str << "Test data size: #{test_set_size}\n"
      str << "Number of mismatched classification: #{num_of_mismatch}\n"
      str << "Classification accuracy: #{accuracy}%\n"
      str << "Iterations of k-means clustering: #{@iteration_counter}\n" if @do_cluster
    end

    result << summary

    File.write("./output.txt", result)
    puts result
    puts "\n=======================================================\n"
    puts "\"output.txt\" is generated."
  end

  private

  def read_training_data
    @training_set = File.readlines(@training_file).reject { |line| line.strip.empty? }.map do |line|
      values = line.split
      IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
    end
  rescue StandardError => e
    puts "Error occurred when reading training data. Exception message: #{e.message}"
  end

  def read_test_data
    @test_set = File.readlines(@test_file).reject { |line| line.strip.empty? }.map do |line|
      values = line.split
      IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
    end
  rescue StandardError => e
    puts "Error occurred when reading test data. Exception message: #{e.message}"
  end

  def classify!(instance)
    k_nearest_neighbours = @training_set.sort_by do |training_instance|
      distance_between(training_instance, instance)
    end.first(k_value)

    instance.classified_class = identify_majority_class(k_nearest_neighbours)
  end

  def distance_between(instance_a, instance_b)
    variance = 0
    variance += ((instance_a.sepal_length - instance_b.sepal_length) / @sepal_length_range) ** 2
    variance += ((instance_a.sepal_width - instance_b.sepal_width) / @sepal_width_range) ** 2
    variance += ((instance_a.petal_length - instance_b.petal_length) / @petal_length_range) ** 2
    variance += ((instance_a.petal_width - instance_b.petal_width) / @petal_width_range) ** 2

    Math.sqrt(variance)
  end

  def identify_majority_class(instances)
    class_count = instances.map(&:label).each_with_object(Hash.new(0)) { |instance, hash| hash[instance] += 1 }
    max_count   = class_count.values.max

    majority_classes = class_count.select { |klass| class_count[klass] == max_count }.keys

    if majority_classes.size > 1
      majority_classes.sample # if more than one majority classes, we randomly choose one
    else
      majority_classes[0]
    end
  end

  def initialise_clusters
    # two options:

    # 1. randomly select num_means(default: 3) initial means
    @clusters = @training_set.sample(@num_clusters).map { |mean| [mean.clone, []] }.to_h

    # 2. use given indexes
    # @clusters = @training_set.values_at(15, 40, 65).map { |mean| [mean.clone, []] }.to_h
  end

  def converge_clusters
    @iteration_counter = 1
    @updated = false

    loop do
      allocate_instances

      # ====== for showing the process of iteration ======
      puts "===================================="
      puts "Iteration: #{@iteration_counter}"

      @clusters.keys.each_with_index do |mean, index|
        puts "Mean #{index + 1}: #{mean}"
        puts "Cluster: #{@clusters[mean].size} instances"
        @clusters[mean].each do |instance|
          puts "  #{instance}"
        end
      end
      # ==================================================

      update_centroids

      break unless @updated

      @iteration_counter += 1
      @clusters.each { |_, cluster| cluster.clear }
    end
  end

  def allocate_instances
    k_means = @clusters.keys

    @training_set.each do |instance|
      closest_mean = k_means.min_by { |mean| distance_between(mean, instance) }
      @clusters[closest_mean] << instance
    end
  end

  def update_centroids
    @updated = false

    @clusters.each do |mean, cluster|
      sepal_length_collection = cluster.map(&:sepal_length)
      sepal_width_collection  = cluster.map(&:sepal_width)
      petal_length_collection = cluster.map(&:petal_length)
      petal_width_collection  = cluster.map(&:petal_width)

      average_sepal_length = sepal_length_collection.reduce(&:+).to_f / sepal_length_collection.size
      average_sepal_width  = sepal_width_collection.reduce(&:+).to_f / sepal_width_collection.size
      average_petal_length = petal_length_collection.reduce(&:+).to_f / petal_length_collection.size
      average_petal_width  = petal_width_collection.reduce(&:+).to_f / petal_width_collection.size

      if (mean.sepal_length - average_sepal_length).abs >= FLOAT_TOLERANCE
        mean.sepal_length = average_sepal_length
        @updated = true
      end

      if (mean.sepal_width - average_sepal_width).abs >= FLOAT_TOLERANCE
        mean.sepal_width = average_sepal_width
        @updated = true
      end

      if (mean.petal_length - average_petal_length).abs >= FLOAT_TOLERANCE
        mean.petal_length = average_petal_length
        @updated = true
      end

      if (mean.petal_width - average_petal_width).abs >= FLOAT_TOLERANCE
        mean.petal_width = average_petal_width
        @updated = true
      end
    end
  end

  def label_training_set_by_clusters # here we label them like "class_1", "class_2", ...
    @clusters.values.each_with_index do |cluster, index|
      cluster.each do |instance|
        instance.clustered_class = "class_#{index + 1}"
      end
    end
  end
end
