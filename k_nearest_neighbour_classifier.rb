class KNearestNeighbourClassifier
  NUM_CLUSTER     = 3
  FLOAT_TOLERANCE = 0.00000001

  attr_reader :k_value, :sepal_length_range, :sepal_width_range, :petal_length_range, :petal_width_range

  def initialize(training_set_file, test_set_file, k_value)
    @training_set_file = training_set_file
    @test_set_file     = test_set_file
    @k_value           = k_value
  end

  def read_data
    read_training_set
    read_test_set
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
    num_of_mismatch = @test_set.count(&:classified_class_mismatched?)
    test_set_size   = @test_set.size
    accuracy        = ((test_set_size - num_of_mismatch).to_f / test_set_size * 100).round(2)

    summary = <<~SUMMARY
      K Value: #{k_value}
      Training data set size: #{@training_set.size}
      Test data set size: #{test_set_size}
      Number of mismatched classification: #{num_of_mismatch}
      Accuracy: #{accuracy}%
    SUMMARY

    result = "=============== Results of classification ===============\n"
    result << ["No.", "sepal_length", "sepal_width", "petal_length", "petal_width", "given_label", "classified_class"].join("  ") << "\n"

    @test_set.each_with_index do |instance, index|
      result << ["#{index + 1}", "#{instance.sepal_length}", "#{instance.sepal_width}", "#{instance.petal_length}", "#{instance.petal_width}", "#{instance.given_label}", "#{instance.classified_class}"].join("  ")
      result << "    # classification mismatched" if instance.classified_class_mismatched?
      result << "\n"
    end

    result << "\n======================= Summary =======================\n"
    result << summary

    File.write("./output.txt", result)
    puts result
    puts "\n=======================================================\n"
    puts "\"output.txt\" is generated."
  end

  private

  def read_training_set
    @training_set = File.readlines(@training_set_file).reject { |line| line.strip.empty? }.map do |line|
      values = line.split
      IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
    end
  rescue Exception => e
    puts "Error occurred when reading training set data. Exception message: #{e.message}"
  end

  def read_test_set
    @test_set = File.readlines(@test_set_file).reject { |line| line.strip.empty? }.map do |line|
      values = line.split
      IrisInstance.new(values[0].to_f, values[1].to_f, values[2].to_f, values[3].to_f, values[4])
    end
  rescue Exception => e
    puts "Error occurred when reading test set data. Exception message: #{e.message}"
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
    class_count = instances.map(&:given_label).each_with_object(Hash.new(0)) { |instance, hash| hash[instance] += 1 }
    max_count   = class_count.values.max

    majority_classes = class_count.select { |klass| class_count[klass] == max_count }.keys

    if majority_classes.size > 1
      majority_classes.sample # if more than one majority classes, we randomly choose one
    else
      majority_classes[0]
    end
  end

  def initialise_clusters
    @training_set.each { |instance| instance.given_label = nil } # remove the label first

    @clusters = @training_set.sample(3).map { |mean| [mean.clone, []] }.to_h
  end

  def converge_clusters
    @iteration_counter  = 1
    @updated = false

    loop do
      allocate_instances
      update_centroids

      if @updated
        @iteration_counter += 1
        @clusters.each { |_, cluster| cluster.clear }
      else
        break
      end
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
        @updated          = true
      end

      if (mean.sepal_width - average_sepal_width).abs >= FLOAT_TOLERANCE
        mean.sepal_width = average_sepal_width
        @updated         = true
      end

      if (mean.petal_length - average_petal_length).abs >= FLOAT_TOLERANCE
        mean.petal_length = average_petal_length
        @updated          = true
      end

      if (mean.petal_width - average_petal_width).abs >= FLOAT_TOLERANCE
        mean.petal_width = average_petal_width
        @updated         = true
      end
    end
  end

  def label_training_set_by_clusters # here we label them like "label_1", "label_2", ...
    @clusters.values.each_with_index do |cluster, index|
      cluster.each do |instance|
        instance.given_label = "label_#{index + 1}"
      end
    end
  end
end
